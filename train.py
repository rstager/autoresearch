"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
import sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

# CHANGED: replaced kernels-based flash attention loader with flash_attn package.
# Original code was:
#   from kernels import get_kernel
#   cap = torch.cuda.get_device_capability()
#   repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
#   fa3 = get_kernel(repo).flash_attn_interface
# and the call site was: fa3.flash_attn_func(q, k, v, ...)
# Reason: kernels-community/flash-attn3 had no compatible build variant for RTX 4090 (sm_89).
# To revert: uninstall flash-attn, restore the above, and remove the q/k/v cast below.
from flash_attn import flash_attn_func

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class BlockConfig:
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768           # block compute/output width; writes x[:, :, :n_embd]
    n_in: int | None = None     # attn input width; None = n_embd; can be set wider for full context
    has_ve: bool = False
    window_size: tuple = (-1, 0)  # (-1, 0) = full context; (k, 0) = sliding window
    enabled: bool = True          # if False, this block is skipped (identity pass-through)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_model: int = 512          # full residual stream width
    blocks: list = field(default_factory=list)  # list[BlockConfig]

    @property
    def n_layer(self):
        return len(self.blocks)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # Slice cos/sin to match this block's head_dim (cos/sin are precomputed for max head_dim)
    y1 = x1 * cos[..., :d] + x2 * sin[..., :d]
    y2 = x1 * (-sin[..., :d]) + x2 * cos[..., :d]
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        n_in = config.n_in if config.n_in is not None else config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(n_in, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_in, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_in, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if config.has_ve else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()  # CHANGED: cast required; flash_attn only accepts fp16/bf16 (F.rms_norm returns fp32)

        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, x_wide, ve, cos_sin, window_size):
        # x      is [B, T, n_embd] — residual stream for this block
        # x_wide is [B, T, n_in]   — wider context for attn (equals x when n_in == n_embd)
        x = x + self.attn(norm(x_wide), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_configs = [deepcopy(bc) for bc in config.blocks]
        self.window_sizes = [bc.window_size for bc in self.block_configs]
        bc0 = self.block_configs[0]
        bc_last = self.block_configs[-1]
        head_dim = max(bc.n_embd // bc.n_head for bc in self.block_configs)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, bc0.n_embd),
            "h": nn.ModuleList([Block(bc) for bc in self.block_configs]),
        })
        pad_size = config.n_model - bc0.n_embd
        self.wte_pad = nn.Parameter(torch.zeros(pad_size)) if pad_size > 0 else None
        self.lm_head = nn.Linear(bc_last.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings (kv_dim is per-block)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, bc.n_kv_head * (bc.n_embd // bc.n_head))
            for i, bc in enumerate(self.block_configs) if bc.has_ve
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        for block, bc in zip(self.transformer.h, self.block_configs):
            s = 3**0.5 * bc.n_embd**-0.5
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = max(bc.n_embd // bc.n_head for bc in self.block_configs)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward), counting only enabled layers."""
        enabled_block_params = sum(
            p.numel()
            for i, block in enumerate(self.transformer.h)
            if self.block_configs[i].enabled
            for p in block.parameters()
        )
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        t = self.config.sequence_len
        attn_flops = 0
        for bc, window_size in zip(self.block_configs, self.window_sizes):
            if not bc.enabled:
                continue
            h = bc.n_head
            q = bc.n_embd // bc.n_head
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (lm_head_params + enabled_block_params) + attn_flops

    def num_scaling_params(self, active_only=False):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        if active_only:
            transformer_matrices = sum(
                p.numel()
                for i, block in enumerate(self.transformer.h)
                if self.block_configs[i].enabled
                for p in block.parameters()
            )
            value_embeds = sum(
                p.numel()
                for k, ve in self.value_embeds.items()
                if self.block_configs[int(k)].enabled
                for p in ve.parameters()
            )
        else:
            transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
            value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.block_configs[0].n_embd
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        wte_pad_params = [self.wte_pad] if self.wte_pad is not None else []
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=wte_pad_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Per-layer Muon groups: one group per (layer_idx, param_shape).
        # The 'active' flag mirrors bc.enabled and is toggled at stage transitions.
        all_layer_muon_params = []
        for layer_idx, (block, bc) in enumerate(zip(self.transformer.h, self.block_configs)):
            layer_params = list(block.parameters())
            all_layer_muon_params.extend(layer_params)
            for shape in sorted({p.shape for p in layer_params}):
                shape_params = [p for p in layer_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=shape_params,
                    layer_idx=layer_idx, active=bc.enabled,
                    lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
        non_muon_params = (lm_head_params + embedding_params + value_embeds_params +
                           wte_pad_params + resid_params + x0_params)
        assert len(list(self.parameters())) == len(non_muon_params) + len(all_layer_muon_params)
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    @staticmethod
    @torch.compile(dynamic=True)
    def _embed(wte, wte_pad, cos, sin, idx):
        B, T = idx.size()
        cos_sin = cos[:, :T], sin[:, :T]
        x = wte(idx)
        if wte_pad is not None:
            pad = wte_pad.expand(B, T, -1)
            x = torch.cat([x, pad], dim=-1)
        x = norm(x)
        return x, cos_sin

    @staticmethod
    @torch.compile(dynamic=True)
    def _block_step(block, resid_lambda, x0_lambda, ve_embed, idx, x, x0, cos_sin, window_size, n_embd, n_in):
        x = resid_lambda * x + x0_lambda * x0
        ve = ve_embed(idx) if ve_embed is not None else None
        x_narrow = x[:, :, :n_embd]
        x_wide = x[:, :, :n_in]
        out = block(x_narrow, x_wide, ve, cos_sin, window_size)
        x = torch.cat([out, x[:, :, n_embd:]], dim=-1)
        return x

    @staticmethod
    @torch.compile(dynamic=True)
    def _head(lm_head, x, targets, n_embd_last):
        x = norm(x)
        logits = lm_head(x[:, :, :n_embd_last])
        logits = logits.float()
        logits = 15 * torch.tanh(logits / 15)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss

    def forward(self, idx, targets=None, reduction='mean'):
        x, cos_sin = self._embed(self.transformer.wte, self.wte_pad, self.cos, self.sin, idx)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            bc = self.block_configs[i]
            if not bc.enabled:
                continue
            n_in = bc.n_in if bc.n_in is not None else bc.n_embd
            ve_embed = self.value_embeds.get(str(i))
            x = self._block_step(block, self.resid_lambdas[i], self.x0_lambdas[i],
                                 ve_embed, idx, x, x0, cos_sin, self.window_sizes[i],
                                 bc.n_embd, n_in)
        if targets is not None:
            return self._head(self.lm_head, x, targets, self.block_configs[-1].n_embd)
        x = norm(x)
        n_embd_last = self.block_configs[-1].n_embd
        logits = self.lm_head(x[:, :, :n_embd_last]).float()
        logits = 15 * torch.tanh(logits / 15)
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                if not group.get('active', True):
                    continue
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Stacked layer training
# ---------------------------------------------------------------------------

def build_stacked_schedule(n_layer, total_matrix_params,
                            total_batch_size=2**19, max_seq_len=2048,
                            initial_device_batch=128, n_top=2,
                            layers_per_stage=1, stage_batch_sizes=None):
    """
    Build the bottom-up stage schedule for stacked layer training.

    The top n_top layers are always live (providing gradient signal to lm_head).
    Stage 0 starts with the first n_top bottom layers + all n_top top layers.
    Each subsequent stage adds layers_per_stage bottom layers working inward.

    Token budget = equal split of 20 * total_matrix_params across all stages.
    Batch size scales down proportionally as more layers are active, keeping
    TOTAL_BATCH_SIZE constant via increased grad_accum_steps.

    Returns list of dicts:
      {new_layers, active_layers, token_budget, device_batch_size, grad_accum_steps}
    """
    # Bottom-up: top n_top layers always live; grow from bottom inward
    top_layers = list(range(n_layer - n_top, n_layer))
    n_bottom = n_layer - n_top
    stages_new = [list(range(n_top)) + top_layers]  # stage 0: first n_top bottom + all top
    for i in range(n_top, n_bottom, layers_per_stage):
        stages_new.append(list(range(i, min(i + layers_per_stage, n_bottom))))
    n_stages = len(stages_new)

    # Valid device batch sizes: divisors of total_batch_size // max_seq_len
    max_divisor = total_batch_size // max_seq_len
    valid_batches = sorted([b for b in range(1, max_divisor + 1) if max_divisor % b == 0],
                           reverse=True)

    token_budget_per_stage = (20 * total_matrix_params // n_stages // total_batch_size) * total_batch_size
    token_budget_per_stage = max(total_batch_size, token_budget_per_stage)

    # Valid device batch sizes: multiples of 16 that divide total_batch_size / max_seq_len
    valid_batches = sorted(
        [b for b in valid_batches if b % 16 == 0 or b <= 16],
        reverse=True
    )
    if not valid_batches:
        valid_batches = [max(b for b in range(1, max_divisor + 1) if max_divisor % b == 0)]

    initial_n_active = len(stages_new[0])
    schedule = []
    active = []
    for stage_idx, new_layers in enumerate(stages_new):
        active = active + new_layers
        n_active = len(active)

        if stage_batch_sizes is not None:
            device_batch = stage_batch_sizes[stage_idx]
        elif stage_idx == 0:
            device_batch = initial_device_batch
        else:
            # Heuristic: scale proportionally to 1/n_active, floor to nearest valid multiple of 16
            target = initial_device_batch * initial_n_active / n_active
            candidates = [b for b in valid_batches if b <= target]
            device_batch = candidates[0] if candidates else valid_batches[-1]

        grad_accum = math.ceil(total_batch_size / (device_batch * max_seq_len))
        schedule.append({
            'new_layers':        list(new_layers),
            'active_layers':     list(active),
            'token_budget':      token_budget_per_stage,
            'device_batch_size': device_batch,
            'grad_accum_steps':  grad_accum,
        })
    return schedule


def calibrate_batch_sizes(schedule, raw_model, optimizer, autocast_ctx, seq_len, total_batch_size,
                           n_probe_steps=5):
    """
    For each stage, find the largest device batch size that doesn't OOM and measure MFU.

    Temporarily enables each stage's active layers on raw_model; restores all-disabled on return.
    Returns list of int batch sizes, one per stage.
    """
    max_batch = total_batch_size // seq_len
    valid_batches = sorted(
        [b for b in range(16, max_batch + 1, 16)],
        reverse=True,
    )
    if not valid_batches:
        valid_batches = [max_batch] if max_batch > 0 else [1]

    device = torch.device("cuda")
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"  GPU VRAM: {total_vram:.1f} GB")
    print(f"  Candidate batch sizes: {valid_batches}")
    calibrated = []

    for stage_idx, s in enumerate(schedule):
        print(f"\n  Stage {stage_idx}/{len(schedule)-1}: layers={s['active_layers']} "
              f"({len(s['active_layers'])} active)")
        # Move all layers to CPU, then bring only active ones to GPU
        for layer_idx in range(len(raw_model.block_configs)):
            raw_model.block_configs[layer_idx].enabled = False
            _move_layer_to(raw_model, optimizer, layer_idx, torch.device("cpu"))
        for layer_idx in s['active_layers']:
            raw_model.block_configs[layer_idx].enabled = True
            _move_layer_to(raw_model, optimizer, layer_idx, device)
        torch.cuda.empty_cache()

        found_batch = None
        for batch_size in valid_batches:
            torch.cuda.empty_cache()
            gc.collect()
            x = y = loss = None
            try:
                x = torch.randint(0, raw_model.config.vocab_size, (batch_size, seq_len), device=device)
                y = torch.randint(0, raw_model.config.vocab_size, (batch_size, seq_len), device=device)
                with autocast_ctx:
                    loss = raw_model(x, y)
                loss.backward()
                raw_model.zero_grad(set_to_none=True)
                found_batch = batch_size
                vram_used = torch.cuda.max_memory_allocated(device) / 1e9
                torch.cuda.reset_peak_memory_stats(device)
                print(f"    batch={batch_size}: OK (peak {vram_used:.1f}/{total_vram:.1f} GB)")
            except torch.cuda.OutOfMemoryError:
                print(f"    batch={batch_size}: OOM")
            finally:
                del x, y, loss
                torch.cuda.empty_cache()
                gc.collect()
            if found_batch is not None:
                break

        if found_batch is None:
            raise RuntimeError(f"Stage {stage_idx}: OOM at all batch sizes {valid_batches}")

        # Measure throughput at found_batch (uncompiled; approximate but consistent)
        num_flops = raw_model.estimate_flops()
        t_times = []
        for _ in range(n_probe_steps + 2):
            x = torch.randint(0, raw_model.config.vocab_size, (found_batch, seq_len), device=device)
            y = torch.randint(0, raw_model.config.vocab_size, (found_batch, seq_len), device=device)
            torch.cuda.synchronize()
            t0 = time.time()
            with autocast_ctx:
                loss = raw_model(x, y)
            loss.backward()
            torch.cuda.synchronize()
            t_times.append(time.time() - t0)
            raw_model.zero_grad(set_to_none=True)
            del x, y, loss
            torch.cuda.empty_cache()

        avg_dt = sum(t_times[2:]) / len(t_times[2:])
        tok_per_sec = found_batch * seq_len / avg_dt
        mfu = 100 * num_flops * tok_per_sec / GPU_BF16_PEAK_FLOPS
        tokens_per_micro = found_batch * seq_len
        grad_accum = math.ceil(total_batch_size / tokens_per_micro)
        print(f"    → batch={found_batch} grad_accum={grad_accum} "
              f"tok/sec={tok_per_sec:,.0f} MFU≈{mfu:.1f}% dt={avg_dt*1000:.0f}ms")
        calibrated.append(found_batch)

    # Restore all-disabled, all-offloaded state for normal training startup
    for layer_idx in range(len(raw_model.block_configs)):
        raw_model.block_configs[layer_idx].enabled = False
        _move_layer_to(raw_model, optimizer, layer_idx, torch.device("cpu"))
    torch.cuda.empty_cache()

    return calibrated


def _move_layer_to(raw_model, optimizer, layer_idx, target_device):
    """Move a transformer layer, its value embeddings, and optimizer state to target_device."""
    block = raw_model.transformer.h[layer_idx]
    block.to(target_device)
    ve_key = str(layer_idx)
    if ve_key in raw_model.value_embeds:
        raw_model.value_embeds[ve_key].to(target_device)
    # Move optimizer state for this layer's param groups
    for group in optimizer.param_groups:
        if group.get('layer_idx') != layer_idx:
            continue
        for p in group['params']:
            if p not in optimizer.state:
                continue
            for k, v in optimizer.state[p].items():
                if isinstance(v, torch.Tensor):
                    optimizer.state[p][k] = v.to(target_device)


def _freeze_layer(raw_model, optimizer, layer_idx):
    """Freeze a layer: disable gradients and delete optimizer state. Layer stays on GPU for inference."""
    block = raw_model.transformer.h[layer_idx]
    for p in block.parameters():
        p.requires_grad_(False)
    ve_key = str(layer_idx)
    if ve_key in raw_model.value_embeds:
        for p in raw_model.value_embeds[ve_key].parameters():
            p.requires_grad_(False)
    # Delete optimizer state and mark groups inactive
    for group in optimizer.param_groups:
        if group.get('layer_idx') != layer_idx:
            continue
        group['active'] = False
        for p in group['params']:
            if p in optimizer.state:
                del optimizer.state[p]
    torch.cuda.empty_cache()
    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  Froze layer {layer_idx} — no grad, optimizer state deleted, GPU mem={vram_used:.1f} GB")


def _offload_inactive_layers(raw_model, optimizer):
    """Move all inactive layers to CPU and mark their optimizer groups inactive."""
    inactive_layers = set()
    for layer_idx, bc in enumerate(raw_model.block_configs):
        if not bc.enabled:
            _move_layer_to(raw_model, optimizer, layer_idx, torch.device("cpu"))
            inactive_layers.add(layer_idx)
    # Mark optimizer groups for inactive layers as inactive
    for group in optimizer.param_groups:
        if group.get('kind') == 'muon' and group.get('layer_idx') in inactive_layers:
            group['active'] = False
    torch.cuda.empty_cache()
    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"Offloaded inactive layers to CPU. GPU memory: {vram_used:.1f} GB")


_layer_activated_at: dict[int, int] = {}  # layer_idx -> stage_idx when first activated


def activate_stage(stage_idx, schedule, raw_model, optimizer, tokenizer):
    """
    Enable new layers for this stage, mark their Muon groups active, and
    return a fresh dataloader with the updated batch size.

    Freezes layers that have been trainable for FREEZE_AFTER_STAGES stages.
    Moves newly active layers (and their optimizer state) from CPU to GPU.
    Muon momentum state is NOT reset — it carries forward across stages.
    """
    t_start = time.time()
    s = schedule[stage_idx]
    new_layer_set = set(s['new_layers'])
    n_layer = len(raw_model.block_configs)
    device = next(p for p in raw_model.lm_head.parameters()).device
    top_layers = set(range(n_layer - N_TOP_LAYERS, n_layer))

    # Freeze layers that have been trainable long enough (skip top layers)
    t0 = time.time()
    for layer_idx in range(n_layer):
        if layer_idx in top_layers:
            continue
        if layer_idx not in _layer_activated_at:
            continue
        stages_active = stage_idx - _layer_activated_at[layer_idx]
        if stages_active >= FREEZE_AFTER_STAGES and raw_model.block_configs[layer_idx].enabled:
            if any(p.requires_grad for p in raw_model.transformer.h[layer_idx].parameters()):
                _freeze_layer(raw_model, optimizer, layer_idx)
    t_freeze = time.time() - t0

    # Move new layers to GPU, enable them, and mark optimizer groups active
    t0 = time.time()
    for layer_idx in s['new_layers']:
        _move_layer_to(raw_model, optimizer, layer_idx, device)
        raw_model.block_configs[layer_idx].enabled = True
        _layer_activated_at[layer_idx] = stage_idx
    for group in optimizer.param_groups:
        if group.get('kind') == 'muon' and group.get('layer_idx') in new_layer_set:
            group['active'] = True
    t_move = time.time() - t0

    t0 = time.time()
    torch.cuda.empty_cache()
    t_cache = time.time() - t0

    t0 = time.time()
    train_loader = make_dataloader(tokenizer, s['device_batch_size'], MAX_SEQ_LEN, "train")
    t_loader = time.time() - t0

    vram_used = torch.cuda.memory_allocated() / 1e9
    trainable = [i for i in range(n_layer) if raw_model.block_configs[i].enabled
                 and any(p.requires_grad for p in raw_model.transformer.h[i].parameters())]
    frozen = [i for i in range(n_layer) if raw_model.block_configs[i].enabled
              and not any(p.requires_grad for p in raw_model.transformer.h[i].parameters())]
    t_total = time.time() - t_start
    print(f"\n[Stage {stage_idx}] new={s['new_layers']}, "
          f"trainable={trainable}, frozen={frozen}, "
          f"batch={s['device_batch_size']}, grad_accum={s['grad_accum_steps']}, "
          f"GPU mem={vram_used:.1f} GB")
    print(f"  timing: freeze={t_freeze:.2f}s move={t_move:.2f}s "
          f"cache={t_cache:.2f}s loader={t_loader:.2f}s total={t_total:.2f}s")
    return train_loader, s['device_batch_size'], s['grad_accum_steps']


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM); used as initial heuristic
N_TOP_LAYERS = 2         # top N layers always live; bottom layers added one per stage
LAYERS_PER_STAGE = 2     # how many bottom layers to add per stage (2 = fewer stages, fewer recompiles)
FREEZE_AFTER_STAGES = 2  # freeze a layer after it's been trainable for this many stages; top layers never freeze
STAGE_BATCH_SIZES: list[int] = []  # calibrated per-stage batch sizes; empty = try import, then calibrate
if not STAGE_BATCH_SIZES:
    try:
        from stage_batch_sizes import STAGE_BATCH_SIZES
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# BF16 peak FLOPS by GPU — used only for MFU reporting, not training logic
_GPU_BF16_PEAK_FLOPS = {
    "H100":  989.5e12,
    "H200":  989.5e12,
    "A100":  312e12,
    "A10G":  70.0e12,
    "L4":    121e12,
    "RTX 4090": 330e12,
    "RTX 4080": 205e12,
    "RTX 3090": 142e12,
    "RTX A6000": 155e12,
}
_gpu_name = torch.cuda.get_device_name(0)
GPU_BF16_PEAK_FLOPS = next((v for k, v in _GPU_BF16_PEAK_FLOPS.items() if k in _gpu_name), 989.5e12)
print(f"GPU: {_gpu_name} — BF16 peak: {GPU_BF16_PEAK_FLOPS/1e12:.1f} TFLOPS")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

S   = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=False, window_size=(1024, 0))
SVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=True,  window_size=(1024, 0))
LVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=True,  window_size=(2048, 0))

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_model=512,
    blocks=[S, SVE, S, LVE, S, SVE, S, LVE],
)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

# Disable all layers initially — activate_stage will enable them in order
for bc in config.blocks:
    bc.enabled = False

param_counts = model.num_scaling_params()
print("Parameter counts (all layers):")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
total_matrix_params = param_counts['transformer_matrices']

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# Start with all layers disabled and offloaded to CPU — activate_stage will move them to GPU
for bc in model.block_configs:
    bc.enabled = False
_offload_inactive_layers(model, optimizer)

# Build stacked schedule; batch sizes come from STAGE_BATCH_SIZES if calibrated, else heuristic
stacked_schedule = build_stacked_schedule(
    n_layer=config.n_layer,
    total_matrix_params=total_matrix_params,
    total_batch_size=TOTAL_BATCH_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    initial_device_batch=DEVICE_BATCH_SIZE,
    n_top=N_TOP_LAYERS,
    layers_per_stage=LAYERS_PER_STAGE,
    stage_batch_sizes=STAGE_BATCH_SIZES if STAGE_BATCH_SIZES else None,
)

if not STAGE_BATCH_SIZES:
    print("\n" + "="*70)
    print("WARNING: No STAGE_BATCH_SIZES found. Running calibration...")
    print("="*70)
    calibrated = calibrate_batch_sizes(
        stacked_schedule, model, optimizer, autocast_ctx, MAX_SEQ_LEN, TOTAL_BATCH_SIZE)
    _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_batch_sizes.py")
    with open(_path, 'w') as _f:
        _f.write(f"STAGE_BATCH_SIZES: list[int] = {calibrated}\n")
    print(f"\nSaved to {_path}")
    print("Run train.py again to start training.")
    sys.exit(0)

print(f"Stacked training: {len(stacked_schedule)} stages, "
      f"token budget per stage: {stacked_schedule[0]['token_budget']/1e6:.0f}M")
for i, s in enumerate(stacked_schedule):
    print(f"  Stage {i}: new={s['new_layers']} active={s['active_layers']} "
          f"batch={s['device_batch_size']} grad_accum={s['grad_accum_steps']}")

current_stage = 0

# Warmup: enable ALL layers, run dummy forward+backward to trigger compilation
# of _embed, _block_step (for all ve/no-ve variants), and _head.
# This ensures no compilation happens mid-training when new layers activate.
print("Compile warmup (all layers)...", flush=True)
_t_compile = time.time()
for bc in model.block_configs:
    bc.enabled = True
for layer_idx in range(len(model.block_configs)):
    _move_layer_to(model, optimizer, layer_idx, device)
_warmup_x = torch.randint(0, model.config.vocab_size, (DEVICE_BATCH_SIZE, MAX_SEQ_LEN), device=device)
_warmup_y = torch.randint(0, model.config.vocab_size, (DEVICE_BATCH_SIZE, MAX_SEQ_LEN), device=device)
with autocast_ctx:
    _warmup_loss = model(_warmup_x, _warmup_y)
_warmup_loss.backward()
model.zero_grad(set_to_none=True)
del _warmup_x, _warmup_y, _warmup_loss
# Restore: disable all, offload, then re-activate stage 0
for bc in model.block_configs:
    bc.enabled = False
_offload_inactive_layers(model, optimizer)
torch.cuda.empty_cache()
print(f"Compile warmup done in {time.time() - _t_compile:.1f}s", flush=True)

# Re-activate stage 0 for training
train_loader, DEVICE_BATCH_SIZE, grad_accum_steps = activate_stage(
    0, stacked_schedule, model, optimizer, tokenizer)

x, y, epoch = next(train_loader)  # prefetch first batch

num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token (stage 0): {num_flops_per_token:e}")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

if wandb is not None:
    try:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "autoresearch"),
            config={
                **asdict(config),
                "total_batch_size": TOTAL_BATCH_SIZE,
                "device_batch_size": DEVICE_BATCH_SIZE,
                "embedding_lr": EMBEDDING_LR,
                "unembedding_lr": UNEMBEDDING_LR,
                "matrix_lr": MATRIX_LR,
                "scalar_lr": SCALAR_LR,
                "weight_decay": WEIGHT_DECAY,
                "adam_betas": ADAM_BETAS,
                "warmup_ratio": WARMUP_RATIO,
                "warmdown_ratio": WARMDOWN_RATIO,
                "final_lr_frac": FINAL_LR_FRAC,
                "num_params_M": num_params / 1e6,
                "grad_accum_steps": grad_accum_steps,
            },
        )
    except Exception as e:
        print(f"wandb init failed (continuing without): {e}")
        wandb = None

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
t_first_token = None  # set on first real training step (excludes compile warmups)
smooth_train_loss = 0
total_training_time = 0
total_tokens_trained = 0  # tokens seen (excludes warmup steps)
tokens_in_stage = 0       # tokens in current stage (excludes warmup steps)
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules (token-based)
    progress = min(total_tokens_trained / (20 * total_matrix_params), 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        if t_first_token is None:
            t_first_token = t0
        total_training_time += dt
        total_tokens_trained += TOTAL_BATCH_SIZE
        tokens_in_stage += TOTAL_BATCH_SIZE

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    global_progress = min(total_tokens_trained / (20 * total_matrix_params), 1.0)
    pct_done = 100 * global_progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / GPU_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} stg{current_stage} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | batch: {DEVICE_BATCH_SIZE} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if wandb is not None and step > 10:
        wandb.log({
            "train/loss": debiased_smooth_loss,
            "train/lr_multiplier": lrm,
            "train/mfu_percent": mfu,
            "train/tok_per_sec": tok_per_sec,
            "train/progress_pct": pct_done,
            "train/stage": current_stage,
            "train/active_layers": len(stacked_schedule[current_stage]['active_layers']),
            "train/device_batch_size": DEVICE_BATCH_SIZE,
            "train/wall_training_seconds": (time.time() - t_first_token) if t_first_token else 0,
            "train/training_seconds": total_training_time,
        }, step=step)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Stage transition: enable next layer pair when this stage's token budget is met
    if (step > 10
            and tokens_in_stage >= stacked_schedule[current_stage]['token_budget']
            and current_stage < len(stacked_schedule) - 1):
        tokens_in_stage = 0
        current_stage += 1
        train_loader, DEVICE_BATCH_SIZE, grad_accum_steps = activate_stage(
            current_stage, stacked_schedule, model, optimizer, tokenizer)
        num_flops_per_token = model.estimate_flops()
        x, y, epoch = next(train_loader)  # prefetch with new loader

    # Token budget exhausted — only stop after warmup steps
    if step > 10 and total_tokens_trained >= 20 * total_matrix_params:
        break

print()  # newline after \r training log

total_tokens = total_tokens_trained

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
wall_training_time = (t_end - t_first_token) if t_first_token else 0
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / GPU_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f} (sum of step dt)")
print(f"wall_training:    {wall_training_time:.1f} (wall clock from first token)")
print(f"total_seconds:    {t_end - t_start:.1f} (includes startup + compile)")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {config.n_layer}")

if wandb is not None:
    wandb.log({
        "eval/val_bpb": val_bpb,
        "eval/peak_vram_mb": peak_vram_mb,
        "eval/mfu_percent": steady_state_mfu,
        "eval/total_tokens_M": total_tokens / 1e6,
        "eval/num_steps": step,
        "eval/training_seconds": total_training_time,
        "eval/wall_training_seconds": wall_training_time,
    })
    wandb.finish()

# Save final checkpoint
ckpt_dir = "/data/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_step{step:05d}.pt")
torch.save({
    "model": model.state_dict(),
    "config": asdict(config),
    "step": step,
    "val_bpb": val_bpb,
    "total_tokens": total_tokens,
}, ckpt_path)
print(f"checkpoint:       {ckpt_path}")
