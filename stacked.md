# Stacked Incremental Layer Training

## Hypothesis

Training a transformer outside-in — enabling outer layers first and progressively adding inner layers — may produce better representations than training all layers simultaneously from scratch. The intuition is that outer layers establish stable input/output representations before inner layers must learn to compose them.

## Method

### Outside-in stage schedule

Layers are enabled in pairs from the outside in: `[0, N-1]`, then `[1, N-2]`, etc. For an odd number of layers the middle layer is enabled last as a singleton. Each stage trains for an equal share of the total Chinchilla-optimal token budget:

```
token_budget_per_stage = 20 * total_matrix_params / n_stages
```

### Batch size scaling

As more layers become active, activation memory grows. Device batch size scales down proportionally to `1/n_active`, floored to the nearest multiple of 16. `TOTAL_BATCH_SIZE` is held constant via increased gradient accumulation steps, so effective learning dynamics stay consistent across stages.

### Optimizer

Muon is restructured with **one param group per (layer_idx, param_shape)** instead of one group per shape across all layers. Each group carries an `active` flag that mirrors `bc.enabled`. Inactive groups are skipped in `optimizer.step()`. Muon momentum is preserved across stage transitions — no reset.

### Inactive layer handling

- `estimate_flops()` counts only enabled layers
- `num_scaling_params(active_only=True)` returns params for enabled layers only
- The FLOPs and param count used for Chinchilla token budgeting reflect the active layer count at each stage

## Implementation

All changes are in `train.py`:
- `build_stacked_schedule(n_layer, total_matrix_params, ...)` — computes the full stage schedule
- `GPT.estimate_flops()` — updated to filter by `bc.enabled`
- `GPT.num_scaling_params(active_only=False)` — new `active_only` flag
- `GPT.setup_optimizer()` — per-layer Muon groups with `active` flag
- `MuonAdamW.step()` — skips groups where `active=False`
- `torch.compile(dynamic=True)` — avoids recompilation at stage transitions when batch size changes

## Status

Implementation complete on the `stacked` branch. Running on `ar-stacked` (GCE us-east1-d, g2-standard-4, 1x L4 spot).
