# Incremental Layer Training Plan

## Hypothesis

Standard training optimizes all layers simultaneously, but early and late layers
may learn more effectively if trained first in isolation before inner layers are
introduced. By training from the outside in, we give the model a chance to
establish stable input/output representations before fitting the intermediate
computation.

## Strategy: Outside-In Staged Training

Train transformer layers in stages, two from each end per stage, working inward.
At each stage transition:
- Newly introduced layers train at full LR
- Previously trained boundary layers continue at reduced LR (10% of full)
- All other layers are frozen (no gradient computation)

### Example: 8-Layer Model

| Stage | Full LR | Reduced LR | Frozen |
|-------|---------|------------|--------|
| 0 | [0, 1, 6, 7] | [] | [2, 3, 4, 5] |
| 1 | [2, 5] | [1, 6] | [0, 3, 4, 7] |
| 2 | [3, 4] | [2, 5] | [0, 1, 6, 7] |

For an N-layer model there are N/2 stages.

## Implementation Details

- Total token budget = Chinchilla-optimal **20 × num_params** (hardware-independent)
- Token budget is divided equally across stages
- Within each stage, the standard warmup/warmdown LR schedule runs against
  the stage's token budget (not the global budget)
- Frozen layers use `requires_grad=False` — PyTorch skips gradient computation
  entirely, saving memory and compute
- Boundary layers (reduced LR) have gradients scaled by `STAGED_BOUNDARY_LR_SCALE=0.1`
  after `loss.backward()`
- Optimizer (Muon) momentum state is reset for newly active layers at each stage
  transition so stale buffers don't corrupt fresh layer training
- Embeddings, lm_head, and scalar params (resid_lambdas, x0_lambdas) always train
  — they are not layer-indexed
- `torch.compile` may trigger a brief recompile (~10-30s) at the first stage
  transition when `requires_grad` changes

## Evaluation

Same as baseline: bits-per-byte (BPB) on fixed validation shard at end of run.
Compare against `autoresearch/mar25` baseline (all layers trained simultaneously)
at same TIME_BUDGET and model config.

## Branch

`staged` — branched from `autoresearch/mar25`
