# Validation: variable-block residual stream

Verify that when `n_model == n_embd` for all blocks and `n_in=None`, the hetero_layers
branch produces the same loss as the unmodified master branch.

## Setup

Both runs use identical:
- random seed (42)
- tokenizer / data shards
- model architecture (same depth, heads, window pattern)
- number of training steps (short smoke run, e.g. 20 steps)
- optimizer hyperparameters

## Step 1 — Baseline: run on master

```bash
git stash          # if any uncommitted changes
git checkout master
uv run train.py 2>&1 | tee /tmp/baseline.log
```

Record the per-step loss values from the log (steps 1–20).

## Step 2 — Run on hetero_layers with equivalent config

```bash
git checkout hetero_layers
```

Confirm the config in `train.py` has `n_model == n_embd` for all blocks and no `n_in` set:

```python
S   = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=False, window_size=(1024, 0))
SVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=True,  window_size=(1024, 0))
LVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=True,  window_size=(2048, 0))

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_model=512,   # == n_embd for all blocks — no wide residual
    blocks=[S, SVE, S, LVE, S, SVE, S, LVE],
)
```

Then run:

```bash
uv run train.py 2>&1 | tee /tmp/hetero.log
```

## Step 3 — Compare

Extract loss values from both logs and diff:

```bash
grep -oP 'loss=\K[\d.]+' /tmp/baseline.log > /tmp/losses_baseline.txt
grep -oP 'loss=\K[\d.]+' /tmp/hetero.log   > /tmp/losses_hetero.txt
paste /tmp/losses_baseline.txt /tmp/losses_hetero.txt
```

## Pass criteria

- Loss values at each step agree to within floating-point tolerance (< 1e-4 absolute
  difference at each step).
- No shape errors, assertion failures, or NaNs in either run.
- Parameter counts (`num_scaling_params`) are identical between branches.

## Notes

- `wte_pad` is a zero vector when `n_model == n_embd`... actually when they are equal,
  `pad_size = 0` so `wte_pad` is `None` and no padding is applied. ✓
- `lm_head` input dim = `bc_last.n_embd = 512` on hetero_layers vs `bc0.n_embd = 512` on
  master — same. ✓
- `c_q/c_k/c_v` input = `n_in = n_embd = 512` — same as master. ✓
