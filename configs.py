"""
Model configuration dataclasses and scaling table for autoresearch experiments.

Adapted from nanochat/scripts/base_train.py build_model_meta() and
nanochat/nanochat/gpt.py _compute_window_sizes() / has_ve().

Usage:
    from configs import BlockConfig, GPTConfig, get_config, CONFIGS
    from dataclasses import replace

    # Build by depth:
    config = get_config(depth=8, vocab_size=8192)

    # Use a named preset (fill in vocab_size at runtime):
    config = replace(CONFIGS["8L-512"], vocab_size=tokenizer.get_vocab_size())

Design rules (from nanochat):
  - model_dim = round_up(depth * aspect_ratio, head_dim)
  - n_head    = model_dim // head_dim
  - n_kv_head = n_head (no GQA by default)
  - window_pattern "SSSL" tiled across layers; final layer always L
      S = short window = ceil(seq_len / 4 / 128) tokens
      L = full context = (-1, 0) in our convention
  - has_ve: alternating, last layer always included
      layer i has VE if: i % 2 == (n_layer - 1) % 2
"""

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Config dataclasses (imported by train.py)
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


# ---------------------------------------------------------------------------
# Scaling parameters (nanochat defaults)
# ---------------------------------------------------------------------------

SEQUENCE_LEN   = 2048
ASPECT_RATIO   = 64    # model_dim = depth * aspect_ratio (then rounded up to head_dim)
HEAD_DIM       = 128   # target attention head dimension
WINDOW_PATTERN = "SSSL"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _short_window(seq_len: int) -> int:
    """Nanochat short window: seq_len/4 rounded up to nearest FA tile (128)."""
    return math.ceil(seq_len / 4 / 128) * 128


def _has_ve(layer_idx: int, n_layer: int) -> bool:
    """True if layer should have a value embedding (alternating, last layer always)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def get_config(
    depth: int,
    vocab_size: int,
    sequence_len: int = SEQUENCE_LEN,
    aspect_ratio: int = ASPECT_RATIO,
    head_dim: int = HEAD_DIM,
    window_pattern: str = WINDOW_PATTERN,
    n_kv_head: int | None = None,
) -> GPTConfig:
    """
    Build a GPTConfig for the given depth using nanochat scaling rules.

    Args:
        depth:          Number of transformer layers.
        vocab_size:     Vocabulary size (from tokenizer). Use 0 as placeholder.
        sequence_len:   Context length.
        aspect_ratio:   model_dim = depth * aspect_ratio (rounded up to head_dim).
        head_dim:       Target head dimension; model_dim rounded to nearest multiple.
        window_pattern: Tiled window pattern string ('SSSL'). S=short, L=full.
        n_kv_head:      KV heads for GQA; defaults to n_head (no GQA).
    """
    base_dim  = depth * aspect_ratio
    model_dim = math.ceil(base_dim / head_dim) * head_dim
    n_head    = model_dim // head_dim
    kv_heads  = n_kv_head if n_kv_head is not None else n_head

    pattern   = window_pattern.upper()
    short_win = _short_window(sequence_len)
    char_to_win = {"S": (short_win, 0), "L": (-1, 0)}

    blocks = []
    for i in range(depth):
        char = pattern[i % len(pattern)]
        win  = char_to_win[char]
        ve   = _has_ve(i, depth)
        if i == depth - 1:   # final layer always full context
            win = (-1, 0)
        blocks.append(BlockConfig(
            n_head=n_head,
            n_kv_head=kv_heads,
            n_embd=model_dim,
            has_ve=ve,
            window_size=win,
        ))

    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_model=model_dim,
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------
# vocab_size=0 is a placeholder — fill in at runtime:
#
#   from configs import CONFIGS
#   from dataclasses import replace
#   config = replace(CONFIGS["8L-512"], vocab_size=tokenizer.get_vocab_size())
#
# Naming: "{depth}L-{model_dim}"
# ---------------------------------------------------------------------------

def _p(depth, aspect_ratio=ASPECT_RATIO, head_dim=HEAD_DIM, window_pattern=WINDOW_PATTERN):
    return get_config(depth=depth, vocab_size=0, aspect_ratio=aspect_ratio,
                      head_dim=head_dim, window_pattern=window_pattern)


CONFIGS: dict[str, GPTConfig] = {
    # --- Nanochat-style scaling table (aspect_ratio=64, head_dim=128) ---
    # dim = round_up(depth * 64, 128)
    "4L-256":   _p(depth=4),    # dim=256,  2 heads,  ~3M params
    "6L-384":   _p(depth=6),    # dim=384,  3 heads, ~11M params
    "8L-512":   _p(depth=8),    # dim=512,  4 heads, ~25M params  ← autoresearch baseline
    "10L-640":  _p(depth=10),   # dim=640,  5 heads, ~49M params
    "12L-768":  _p(depth=12),   # dim=768,  6 heads, ~85M params
    "16L-1024": _p(depth=16),   # dim=1024, 8 heads, ~201M params
    "20L-1280": _p(depth=20),   # dim=1280, 10 heads, ~393M params

    # --- Deeper/narrower (same param budget as baseline, more layers) ---
    "12L-384":  _p(depth=12, aspect_ratio=32),   # dim=384, 3 heads, ~26M params
    "16L-384":  _p(depth=16, aspect_ratio=24),   # dim=384, 3 heads, ~28M params (very deep/narrow)

    # --- Shallower/wider (same param budget as baseline, fewer layers) ---
    "4L-512":   _p(depth=4,  aspect_ratio=128),  # dim=512, 4 heads, ~13M params
    "6L-640":   _p(depth=6,  aspect_ratio=128),  # dim=768→640..., uses head_dim rounding
}


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarize_configs():
    """Print a table of all named presets with key stats."""
    print(f"{'Name':<14} {'depth':>6} {'dim':>6} {'n_head':>7} {'params_M':>9} {'short_win':>10}  {'ve_layers'}")
    print("-" * 72)
    for name, cfg in CONFIGS.items():
        depth  = cfg.n_layer
        dim    = cfg.n_model
        n_head = cfg.blocks[0].n_head if cfg.blocks else 0
        params = 12 * depth * dim**2 / 1e6
        short  = cfg.blocks[0].window_size[0] if cfg.blocks else -1
        ve     = [i for i, b in enumerate(cfg.blocks) if b.has_ve]
        print(f"{name:<14} {depth:>6} {dim:>6} {n_head:>7} {params:>9.1f} {short:>10}  {ve}")


if __name__ == "__main__":
    summarize_configs()
