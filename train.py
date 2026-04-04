"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from datetime import datetime
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
class DeltaNetConfig:
    """Config for a DeltaNet linear-attention block (drop-in replacement for BlockConfig)."""
    n_head: int = 4
    n_embd: int = 512           # block compute/output width; writes x[:, :, :n_embd]
    n_in: int | None = None     # input width; None = n_embd
    expand_v: int = 2           # value head_dim multiplier (head_dim_v = expand_v * head_dim)
    chunk_size: int = 64        # chunk size for parallel delta scan
    has_ve: bool = False        # not used; present for interface compatibility
    window_size: tuple = (-1, 0)  # not used; present for interface compatibility
    enabled: bool = True


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


# ---------------------------------------------------------------------------
# DeltaNet linear attention
# ---------------------------------------------------------------------------

def _delta_scan_chunk(q, k, v, beta, chunk_size):
    """
    Parallel delta-rule scan over one chunk.

    Args:
        q, k : [B, h, C, d]     (C = chunk_size, d = head_dim)
        v    : [B, h, C, d_v]   (d_v = expand_v * d)
        beta : [B, h, C, 1]     scalar forget gate per position

    Returns:
        o    : [B, h, C, d_v]   output for this chunk
        S    : [B, h, d, d_v]   updated state at end of chunk
    """
    B, h, C, d = q.shape
    d_v = v.shape[-1]
    dtype = q.dtype
    S = torch.zeros(B, h, d, d_v, device=q.device, dtype=torch.float32)

    # Sequential loop within the chunk (C is small, typically 64).
    # Each step: S = (1-beta)*S + beta*(k_t^T @ v_t)
    #            o_t = q_t @ S
    outputs = []
    for t in range(C):
        kt = k[:, :, t, :].unsqueeze(-1).float()   # [B,h,d,1]
        vt = v[:, :, t, :].unsqueeze(-2).float()   # [B,h,1,d_v]
        bt = beta[:, :, t, :].float()              # [B,h,1]
        S = (1.0 - bt.unsqueeze(-1)) * S + bt.unsqueeze(-1) * (kt * vt)
        qt = q[:, :, t, :].unsqueeze(-2).float()   # [B,h,1,d]
        ot = (qt @ S).squeeze(-2)                  # [B,h,d_v]
        outputs.append(ot)
    o = torch.stack(outputs, dim=2).to(dtype)      # [B,h,C,d_v]
    return o, S


class DeltaNetAttention(nn.Module):
    """
    DeltaNet: causal linear attention with per-step delta update rule.

    State update:  S_t = (1 - beta_t) * S_{t-1} + beta_t * (k_t ⊗ v_t)
    Output:        o_t = q_t @ S_t

    Uses chunked parallel scan for training efficiency.
    Reference: Schlag et al. 2021 "Linear Transformers Are Secretly Fast Weight Programmers"
               + DeltaNet (Yang et al. 2024)
    """

    def __init__(self, config):
        super().__init__()
        assert isinstance(config, DeltaNetConfig)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.chunk_size = config.chunk_size
        n_in = config.n_in if config.n_in is not None else config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.head_dim_v = self.head_dim * config.expand_v
        assert config.n_embd % config.n_head == 0

        self.c_q    = nn.Linear(n_in, self.n_head * self.head_dim, bias=False)
        self.c_k    = nn.Linear(n_in, self.n_head * self.head_dim, bias=False)
        self.c_v    = nn.Linear(n_in, self.n_head * self.head_dim_v, bias=False)
        self.c_beta = nn.Linear(n_in, self.n_head, bias=False)
        self.c_proj = nn.Linear(self.n_head * self.head_dim_v, self.n_embd, bias=False)

    def forward(self, x, ve, cos_sin, window_size):
        # ve and window_size are unused — present for interface compatibility
        B, T, _ = x.shape
        C = self.chunk_size
        assert T % C == 0, f"Sequence length {T} must be divisible by chunk_size {C}"
        n_chunks = T // C

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim_v)

        # Normalise q, k (same as standard attention path)
        q, k = norm(q), norm(k)

        # beta: per-head forget gate in (0,1)
        beta = torch.sigmoid(self.c_beta(x))  # [B, T, n_head]
        beta = beta.unsqueeze(-1)              # [B, T, n_head, 1]

        # Reshape to [B, h, n_chunks, C, d]
        q    = q.permute(0, 2, 1, 3).reshape(B, self.n_head, n_chunks, C, self.head_dim)
        k    = k.permute(0, 2, 1, 3).reshape(B, self.n_head, n_chunks, C, self.head_dim)
        v    = v.permute(0, 2, 1, 3).reshape(B, self.n_head, n_chunks, C, self.head_dim_v)
        beta = beta.permute(0, 2, 1, 3).reshape(B, self.n_head, n_chunks, C, 1)

        # Process chunks sequentially (state S carried across chunks)
        all_outputs = []
        S = torch.zeros(B, self.n_head, self.head_dim, self.head_dim_v,
                        device=x.device, dtype=x.dtype)
        for ci in range(n_chunks):
            o_chunk, S = _delta_scan_chunk(
                q[:, :, ci], k[:, :, ci], v[:, :, ci], beta[:, :, ci], C
            )
            S = S.to(x.dtype)
            all_outputs.append(o_chunk)  # [B, h, C, d_v]

        # [B, h, T, d_v] -> [B, T, h*d_v]
        out = torch.cat(all_outputs, dim=2)
        out = out.permute(0, 2, 1, 3).reshape(B, T, self.n_head * self.head_dim_v)
        return self.c_proj(out)


class DeltaNetBlock(nn.Module):
    """DeltaNet block — same external interface as Block."""

    def __init__(self, config):
        super().__init__()
        assert isinstance(config, DeltaNetConfig)
        self.attn = DeltaNetAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, x_wide, ve, cos_sin, window_size):
        x = x + self.attn(norm(x_wide), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


def make_block(config):
    """Factory: return the right block class for the given config type."""
    if isinstance(config, DeltaNetConfig):
        return DeltaNetBlock(config)
    return Block(config)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_configs = list(config.blocks)
        self.window_sizes = [bc.window_size for bc in self.block_configs]
        bc0 = self.block_configs[0]
        bc_last = self.block_configs[-1]
        sa_configs = [bc for bc in self.block_configs if not isinstance(bc, DeltaNetConfig)]
        head_dim = max(bc.n_embd // bc.n_head for bc in sa_configs) if sa_configs else 0
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, bc0.n_embd),
            "h": nn.ModuleList([make_block(bc) for bc in self.block_configs]),
        })
        pad_size = config.n_model - bc0.n_embd
        self.wte_pad = nn.Parameter(torch.zeros(pad_size)) if pad_size > 0 else None
        self.lm_head = nn.Linear(bc_last.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings (kv_dim is per-block; DeltaNetConfig blocks don't support has_ve)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, bc.n_kv_head * (bc.n_embd // bc.n_head))
            for i, bc in enumerate(self.block_configs)
            if bc.has_ve and not isinstance(bc, DeltaNetConfig)
        })
        # Rotary embeddings (only needed for standard attention blocks)
        self.rotary_seq_len = config.sequence_len * 10
        if head_dim > 0:
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        else:
            cos = sin = torch.zeros(1)
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
            if isinstance(bc, DeltaNetConfig):
                torch.nn.init.zeros_(block.attn.c_beta.weight)
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
            if isinstance(block, Block) and block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings (only for standard attention blocks)
        sa_configs = [bc for bc in self.block_configs if not isinstance(bc, DeltaNetConfig)]
        if sa_configs:
            head_dim = max(bc.n_embd // bc.n_head for bc in sa_configs)
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
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        wte_pad_numel = self.wte_pad.numel() if self.wte_pad is not None else 0
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          wte_pad_numel + self.resid_lambdas.numel() + self.x0_lambdas.numel())
        t = self.config.sequence_len
        attn_flops = 0
        for bc, window_size in zip(self.block_configs, self.window_sizes):
            h = bc.n_head
            q = bc.n_embd // bc.n_head
            if isinstance(bc, DeltaNetConfig):
                # Linear attention: O(T * d^2) per head, no quadratic term
                attn_flops += 4 * h * q * q * t  # approx for state ops
            else:
                window = window_size[0]
                effective_seq = t if window < 0 else min(window, t)
                attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.block_configs[0].n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        wte_pad_params = [self.wte_pad] if self.wte_pad is not None else []
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) +
            len(wte_pad_params))
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
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)          # [B, T, bc0.n_embd]
        if self.wte_pad is not None:
            pad = self.wte_pad.expand(B, T, -1)
            x = torch.cat([x, pad], dim=-1)    # [B, T, n_model]
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            bc = self.block_configs[i]
            if not bc.enabled:
                continue
            n_embd = bc.n_embd
            n_in = bc.n_in if bc.n_in is not None else n_embd
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x_narrow = x[:, :, :n_embd]
            x_wide = x[:, :, :n_in]
            out = block(x_narrow, x_wide, ve, cos_sin, self.window_sizes[i])  # [B,T,n_embd]
            x = torch.cat([out, x[:, :, n_embd:]], dim=-1)
        x = norm(x)

        softcap = 15
        n_embd_last = self.block_configs[-1].n_embd
        logits = self.lm_head(x[:, :, :n_embd_last])
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
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
                self._step_muon(group)

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

DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# DeltaNet early layers (linear attention, simulates fast-weight memory)
DN  = DeltaNetConfig(n_head=4, n_embd=512, expand_v=2, chunk_size=64)

# Standard attention layers
S   = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=False, window_size=(1024, 0))
SVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=True,  window_size=(1024, 0))
LVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=True,  window_size=(2048, 0))

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_model=512,
    # First 2 layers: DeltaNet (linear attention, O(T) memory, good for early feature extraction)
    # Remaining 6 layers: standard attention (as in hetero_layers baseline)
    blocks=[DN, DN, S, LVE, S, SVE, S, LVE],
)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

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
smooth_train_loss = 0
total_training_time = 0
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

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
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
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if wandb is not None and step > 10:
        wandb.log({
            "train/loss": debiased_smooth_loss,
            "train/lr_multiplier": lrm,
            "train/mfu_percent": mfu,
            "train/tok_per_sec": tok_per_sec,
            "train/progress_pct": pct_done,
        }, step=step)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
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
    })
    wandb.finish()

# Save final checkpoint
ckpt_dir = "/data/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_step{step:05d}.pt")
torch.save({
    "model": model._orig_mod.state_dict(),
    "config": asdict(config),
    "step": step,
    "val_bpb": val_bpb,
    "total_tokens": total_tokens,
}, ckpt_path)
print(f"checkpoint:       {ckpt_path}")
