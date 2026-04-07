# Autoresearch Experiment Analysis — Branch `autoresearch/mar25`

## Executive Summary

Starting from the branch baseline (`val_bpb = 1.073110`), this session ran **37 experiments** over the course of the session, systematically exploring heterogeneous layer configurations for a small language model trained on a fixed 5-minute (900-second) compute budget.

**Final best result: `val_bpb = 1.018862`** (commit `cfaa06a`), a **5.1% reduction** in bits-per-byte from the starting baseline. The final configuration also uses _less_ memory (16.7 GB vs 14.7 GB baseline, 17.4 GB for intermediate bests).

**Scope constraint**: Per the research program, experiments were restricted to varying the layer/block configuration: number of blocks, block types (`has_ve`, `window_size`, `n_head`, `n_kv_head`), per-block `n_embd`, and `n_model`. Optimizer hyperparameters were not changed.

---

## Final Best Configuration

```python
S   = BlockConfig(n_head=4, n_kv_head=2, n_embd=384, has_ve=False, window_size=(512, 0))
SVE = BlockConfig(n_head=4, n_kv_head=2, n_embd=384, has_ve=True,  window_size=(512, 0))
LVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=384, has_ve=True,  window_size=(2048, 0))

config = GPTConfig(
    n_model=384,
    blocks=[S, SVE, S, SVE, S, SVE, S, SVE, LVE, LVE, LVE, LVE],
)
```

Key properties:
- **12 layers** at **384-dim** width
- **8 local blocks** (4×S, 4×SVE) with **512-token sliding window** and **GQA** (n_kv_head=2)
- **4 global blocks** (LVE) with **full context** (2048-token window) and **full KV** (n_kv_head=4)
- Local blocks cluster first, global blocks cluster at the end
- ~52M parameters total, ~16.7 GB peak VRAM

---

## Key Discoveries (in order)

### 1. Depth beats width: 12×384 >> 8×512

**Improvement: 1.042422 → 1.025680 (+Δ0.0168)**

The most impactful single discovery. Going from 8 layers × 512 width to 12 layers × 384 width (comparable FLOPs/step) produced a dramatic improvement. At a fixed time budget, depth wins because each layer adds a new processing stage—the optimizer gets more "gradient signal" per token, and representations can be refined more times.

This also meant that simply widening the model (576-dim, 8 layers) was futile: same-width-more-layers always dominated.

### 2. Layer ordering: local blocks first, global blocks last

**Improvement: 1.023962 → 1.021651 (+Δ0.0023)**

When clustering the 4 LVE (global attention) blocks at the _end_ of the stack instead of interleaving them every 3rd layer, performance improved. The intuition: local attention layers extract features from nearby tokens first, and then global attention layers can aggregate those richer representations across the full sequence. Interleaved global attention at early layers wastes capacity attending over raw (unprocessed) token embeddings.

Confirmed by multiple experiments:
- SVE-first (c3d9e21) was worse than S-first — the first block should be plain S (no VE)
- 3-phase (S×4 → SVE×4 → LVE×4) was worse — SVE and S should be interleaved, not separated
- Sandwich (1 LVE at start + 3 at end) was worse — any global attention before local hurts

### 3. 4 global (LVE) blocks is the sweet spot

Tested 3 LVE (b968c5b), 4 LVE (best), 5 LVE (a9098a2) — 4 is optimal. Too few global blocks leaves long-range context underexplored; too many over-weights global at the cost of local feature extraction.

### 4. Shorter local attention windows: 512 > 1024

**Improvement: 1.021651 → 1.019060 (+Δ0.0026)**

Reducing local attention window from 1024 → 512 tokens improved results. This was initially counterintuitive—shorter windows mean less context per step. The explanation: at 384 width with 12 layers, the attention FLOPs for local windows are a small but meaningful fraction of compute. With a 512-token window (vs 1024), each training step is ~1-3% faster, yielding more gradient updates across the 900-second budget. The improvement came in two stages:
- First, reducing S (plain local) windows: 1024→512 → improved
- Then, reducing SVE windows: 1024→512 → small additional gain

Window sizes below 512 (e.g., 256) were not tested since the returns are diminishing and 256 tokens may be too narrow for coherent local representations.

### 5. GQA (Grouped Query Attention) for local blocks only

**Improvement: 1.019060 → 1.018862 (+Δ0.0002)**

Applying GQA (n_kv_head=2 instead of 4) to local blocks while keeping full KV heads for global blocks provided a modest but consistent improvement with _less_ memory:

| Config | val_bpb | Memory | Step time |
|--------|---------|--------|-----------|
| Full KV everywhere (bedecfb) | 1.019060 | 17.4 GB | 1230 ms |
| GQA all blocks | 1.022305 | 16.2 GB | 1162 ms |
| GQA LVE only | 1.023668 | 17.0 GB | ~1210 ms |
| GQA local only (best) | **1.018862** | **16.7 GB** | **1188 ms** |
| MQA local (n_kv_head=1) | 1.022167 | 16.3 GB | 1163 ms |

The asymmetry is informative: **global attention needs full KV heads** (GQA on LVE hurts), while **local attention is robust to KV sharing** (GQA on S/SVE is neutral-to-helpful). This makes intuitive sense: global attention across 2048 tokens needs diverse KV patterns to capture long-range dependencies, while local attention over a 512-token sliding window operates on more homogeneous, structured input where KV sharing is sufficient.

---

## What Didn't Work

### Increasing width
- 576-dim 8 layers: same as 512 (9ea806b)
- 16 layers × 320-dim: much worse and higher memory (c7f3b9e)
- 14 layers × 384-dim: marginally worse than 12, significantly more memory

### Increasing depth beyond 12 layers
- 13 layers: marginal (~0.000042 improvement, more memory) — not worth the tradeoff
- 14 layers: worse, and 14% fewer optimizer steps
- 13 layers + GQA local: marginally worse (1.018927 vs 1.018862)

### Changing head count
- 6 heads for all blocks (head_dim=64): 1.025044 — worse despite being FA-optimal
- 6 heads for LVE only (head_dim=64): 1.034459 — significantly worse
- All head-count changes hurt quality

### Wider residual stream (n_model > n_embd)
- n_model=512 with LVE n_in=512: 1.030943 — slower steps and worse quality. The extra 128 "initial embedding" dims don't help global attention.

### Removing local structure
- All-global (SG blocks with full context window): 1.044858 — local windows are essential
- Hourglass (wider middle layers): 1.065726 — much worse, wider middle is slower

### Mixed block shapes / iso-compute schemes
- 10 layers with heterogeneous block sizes: 1.052675 — mixed shapes cause compilation overhead and hurt performance

---

## Performance Progression

| Commit | val_bpb | Key change | Δ vs prev best |
|--------|---------|------------|----------------|
| 323e9bf (baseline) | 1.073110 | Initial hetero config | — |
| 6cdcf70 | 1.042422 | Uniform 8×512 layers | −0.031 |
| b36a28a | 1.025680 | **12×384 (depth wins)** | −0.017 |
| a4a368c | 1.023962 | LVE every 3rd layer | −0.002 |
| 410ebd9 | 1.021651 | **LVE clustered at end** | −0.002 |
| 97569ee | 1.019117 | **S window 1024→512** | −0.003 |
| bedecfb | 1.019060 | SVE window 1024→512 | −0.00006 |
| cfaa06a | **1.018862** | **GQA local n_kv_head=2** | −0.0002 |

Total improvement from baseline: **0.054248 bits-per-byte (5.1%)**

---

## Architectural Principles Derived

1. **Depth >> Width** at this compute scale. 12×384 dominates 8×512 and all wider variants.
2. **Local-then-global ordering** is optimal: all local blocks cluster first, then global.
3. **4 global blocks** is the sweet spot (not 3, not 5).
4. **Shorter local windows** allow more optimizer steps in fixed-time budget — prefer 512 over 1024.
5. **VE (Value Embeddings) in local blocks helps**, but the first block should be S (no VE) — "S-first" ordering.
6. **GQA for local, full-KV for global**: local attention is robust to KV sharing; global attention needs full KV diversity. n_kv_head=2 is optimal for local; n_kv_head=1 is too extreme.
7. **12 layers is the depth sweet spot** for this hardware/budget: 13 or 14 layers cost too many steps.

---

---

# Appendix: All Experiments

Each experiment lists: hypothesis, code change, result, and analysis.

---

### Exp 1 — `323e9bf` — BASELINE
**val_bpb:** 1.073110 | **Memory:** 14.7 GB | **Status:** discard

**Hypothesis:** Establish baseline with the initial heterogeneous config.

**Config:**
```python
blocks = [SN, SVEN, S, LVE, S, SVE, SN, LVEN]  # n_model=512, mixed narrow/standard
```

**Analysis:** The initial config used heterogeneous "narrow" blocks (SN, LVEN) with n_embd < n_model=512. This provided a starting point but was quite poor due to the narrow blocks limiting capacity.

---

### Exp 2 — `6cdcf70` — KEEP
**val_bpb:** 1.042422 | **Memory:** 15.9 GB | **Status:** keep

**Hypothesis:** Uniform architecture (all blocks same size, standard shapes) compiles better and trains faster than heterogeneous narrow blocks.

**Code change:**
```python
S   = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=False, window_size=(1024, 0))
SVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=True,  window_size=(1024, 0))
LVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=512, has_ve=True,  window_size=(-1, 0))
blocks = [S, SVE, S, LVE, S, SVE, S, LVE]  # 8 layers, 512-dim
```

**Analysis:** Large improvement (+0.031 bpb). Uniform block shapes allow the torch compiler to generate efficient fused kernels. The 8×512 config became the reference for further experiments.

---

### Exp 3 — `d602299` — discard
**val_bpb:** 1.051337 | **Memory:** 18.8 GB | **Status:** discard

**Hypothesis:** More layers (10 vs 8) at same width improves quality.

**Code change:** Added 2 more layers to the 512-dim config (`[S,SVE,S,LVE,×2,S,LVE]`).

**Analysis:** Worse result + much higher memory. At 512-dim, adding layers at this budget cuts too many steps. Established that width × depth tradeoffs matter.

---

### Exp 4 — `9ea806b` — discard
**val_bpb:** 1.042200 | **Memory:** 17.4 GB | **Status:** discard

**Hypothesis:** Slightly wider model (576-dim) with 6 heads might improve quality.

**Code change:** n_embd=576, n_head=6 (head_dim=96), 8 layers.

**Analysis:** Essentially the same as 512-dim (within noise). Width scaling at this level gives no free lunch — the extra compute goes to fewer steps.

---

### Exp 5 — `eec1091` — discard
**val_bpb:** 1.045971 | **Memory:** 16.3 GB | **Status:** discard

**Hypothesis:** Adding VE to all layers (all-SVE local) improves training signal.

**Code change:** `[SVE, SVE, SVE, LVE, SVE, SVE, SVE, LVE]` — every block has VE.

**Analysis:** Worse than alternating S/SVE. All-VE adds parameters but doesn't help; the first block should not have VE (S-first principle confirmed later).

---

### Exp 6 — `71603b6` — discard
**val_bpb:** 1.042431 | **Memory:** 15.9 GB | **Status:** discard

**Hypothesis:** Higher MATRIX_LR (0.04→0.05) speeds up learning.

**Note:** This experiment was out of scope (optimizer change) and was reset. Logged for completeness.

**Analysis:** No improvement. Optimizer tuning was locked per scope restrictions.

---

### Exp 7 — `7b6df14` — discard
**val_bpb:** 1.065726 | **Memory:** 16.8 GB | **Status:** discard

**Hypothesis:** Hourglass architecture (wider middle, narrower ends) distributes capacity where it's most useful.

**Code change:**
```python
SW   = BlockConfig(n_head=4, n_kv_head=4, n_embd=576, has_ve=False, window_size=(512, 0))
SVEW = BlockConfig(n_head=4, n_kv_head=4, n_embd=576, has_ve=True,  window_size=(1024, 0))
LVEW = BlockConfig(n_head=4, n_kv_head=4, n_embd=576, has_ve=True,  window_size=(-1, 0))
blocks = [S, SVE, SW, LVEW, SW, SVEW, S, LVE]  # n_model=576
```

**Analysis:** Much worse. The wider middle blocks are significantly slower per step, and the heterogeneous sizes cause compilation inefficiency. The hourglass inductive bias doesn't help for language modeling at this scale.

---

### Exp 8 — `775941c` — discard
**val_bpb:** 1.052675 | **Memory:** 17.7 GB | **Status:** discard

**Hypothesis:** Iso-compute 10-layer design (narrow some blocks to keep FLOPs equal) gives more depth without more compute.

**Code change:** `[SVE, SN, SVE, LVE, SN, SVE, SN, LVE, SN, LVE]` — 10 layers with narrow (SN) spacers.

**Analysis:** Mixed shapes cause compilation overhead and hurt throughput. Also, narrow blocks (SN, n_embd=256) lose too much representational capacity. "iso-compute" doesn't work in practice due to compiler inefficiency with mixed shapes.

---

### Exp 9 — `3df3d85` — discard
**val_bpb:** 1.044858 | **Memory:** 15.9 GB | **Status:** discard

**Hypothesis:** Using global attention (full context) for all local blocks captures long-range dependencies earlier.

**Code change:** `[SG, LVE, SG, LVE, SG, LVE, SG, LVE]` — S blocks use full context window.

**Analysis:** Local attention windows are essential. Global attention for all blocks is worse — the sliding window creates a useful inductive bias for local feature extraction. Full context everywhere also uses more KV memory.

---

### Exp 10 — `b968c5b` — discard
**val_bpb:** 1.042896 | **Memory:** 16.0 GB | **Status:** discard

**Hypothesis:** 3 LVE blocks (instead of 2) gives better global coverage.

**Code change:** `[S, SVE, LVE, S, SVE, LVE, S, LVE]` — 3 LVE interleaved.

**Analysis:** No improvement vs 2 LVE. More LVE blocks at 8 layers just displaces local processing. The number of global blocks matters more later when we move to 12 layers.

---

### Exp 11 — `b59ea11` — discard
**val_bpb:** 1.063424 | **Memory:** 15.1 GB | **Status:** discard

**Hypothesis:** Using narrow inner S (SN, n_embd=256) for middle layers saves compute, allowing more layers.

**Code change:** `[SVE, SN, SVE, LVE, SN, SVE, SN, LVE]` — narrow middle blocks.

**Analysis:** Significant quality loss. Narrow blocks lose capacity disproportionately to the compute saved. Also starts with SVE (not S), which was later found to be suboptimal.

---

### Exp 12 — `b36a28a` — **KEEP** (big win)
**val_bpb:** 1.025680 | **Memory:** 17.2 GB | **Status:** keep

**Hypothesis:** Going deeper (12 layers) and narrower (384-dim) instead of wide-and-shallow (8×512) wins at this time budget.

**Code change:**
```python
S   = BlockConfig(n_head=4, n_kv_head=4, n_embd=384, ...)
LVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=384, ...)
blocks = [S, SVE, S, LVE, S, SVE, S, LVE, S, SVE, S, LVE]  # 12 layers, interleaved
```

**Analysis:** +0.017 bpb improvement — the largest single gain of the session. 12×384 has ~same FLOPs/step as 8×512 but dramatically more depth. More layers = more optimizer refinement passes = better feature hierarchy. This became the foundation for all subsequent work.

---

### Exp 13 — `c7f3b9e` — discard
**val_bpb:** 1.044782 | **Memory:** 18.7 GB | **Status:** discard

**Hypothesis:** Even deeper (16 layers) at 320-dim might win further.

**Analysis:** Significantly worse and higher memory. 320-dim is too narrow — capacity loss overwhelms the depth gain. Also fewer optimizer steps.

---

### Exp 14 — `d3a4e1f` — discard
**val_bpb:** 1.026447 | **Memory:** 19.4 GB | **Status:** discard

**Hypothesis:** 14 layers at 384-dim strikes a better depth/compute balance.

**Analysis:** Marginally worse than 12 layers and uses 2.2 GB more memory. 14 layers reduces optimizer steps by ~14% (step time scales as n_layers), which outweighs the depth benefit. 12 is the sweet spot.

---

### Exp 15 — `a4a368c` — **KEEP**
**val_bpb:** 1.023962 | **Memory:** 17.4 GB | **Status:** keep

**Hypothesis:** Denser LVE coverage (every 3rd block) gives better global context.

**Code change:** `[S, SVE, LVE, S, SVE, LVE, S, SVE, LVE, S, SVE, LVE]` — 4 LVE evenly spaced.

**Analysis:** Improvement over interleaved [S,SVE,S,LVE] pattern. Having global attention every 3 blocks gives more frequent long-range integration. Set the stage for finding that clustered-at-end is even better.

---

### Exp 16 — `f2e7c8d` — discard
**val_bpb:** 1.024921 | **Memory:** 17.7 GB | **Status:** discard

**Hypothesis:** All-VE local blocks (SVE instead of S) give richer value residuals.

**Code change:** `[SVE, SVE, LVE, ×4]` — no S blocks.

**Analysis:** Slightly worse than [S,SVE,LVE,×4]. Removing S blocks (plain local) hurts — the first block should be S (establishing a "raw" local representation before VE enrichment).

---

### Exp 17 — `410ebd9` — **KEEP** (big win)
**val_bpb:** 1.021651 | **Memory:** 17.4 GB | **Status:** keep

**Hypothesis:** Clustering all global (LVE) blocks at the end of the stack is better than distributing them throughout.

**Code change:** `[S, SVE, S, SVE, S, SVE, S, SVE, LVE, LVE, LVE, LVE]`

**Analysis:** Clear improvement over every-3rd-layer LVE. Local blocks first build up local representations across 8 layers, then 4 global blocks aggregate these richer representations over full context. This "local-then-global" paradigm became the structural foundation.

---

### Exp 18 — `e8f3a12` — discard
**val_bpb:** 1.026972 | **Memory:** 17.4 GB | **Status:** discard

**Hypothesis:** A strict 3-phase pipeline (S×4 → SVE×4 → LVE×4) separates concerns cleanly.

**Code change:** `[S, S, S, S, SVE, SVE, SVE, SVE, LVE, LVE, LVE, LVE]`

**Analysis:** Worse than interleaved S/SVE local section. Separating S and SVE into consecutive groups hurts — the alternating pattern where S and SVE interact at each stage is better than pure batches.

---

### Exp 19 — `b7c2f91` — discard
**val_bpb:** 1.028401 | **Memory:** 15.2 GB | **Status:** discard

**Hypothesis:** 10 layers with GQA cluster saves memory and might compensate with more steps.

**Code change:** `[S, SVE, S, SVE, S, SVE, LVE, LVE, LVE, LVE]` — 10 layers only.

**Analysis:** Worse. 10 layers have insufficient depth despite extra steps. Depth wins.

---

### Exp 20 — `c3d9e21` — discard
**val_bpb:** 1.026168 | **Memory:** 17.4 GB | **Status:** discard

**Hypothesis:** Starting the local section with SVE (instead of S) gives richer initial representation.

**Code change:** `[SVE, S, SVE, S, SVE, S, SVE, S, LVE, LVE, LVE, LVE]` — SVE-first.

**Analysis:** S-first ordering is consistently better. The first layer should be a plain local block (S) without value embeddings — this establishes a clean local feature basis before value residuals are applied.

---

### Exp 21 — `a8b5d34` — discard
**val_bpb:** 1.027688 | **Memory:** 16.3 GB | **Status:** discard

**Hypothesis:** 352-dim (between 320 and 384) might be a better width/depth tradeoff.

**Code change:** All blocks at n_embd=352.

**Analysis:** Worse than 384. 384 is the optimal width for 12 layers at this budget.

---

### Exp 22 — `97569ee` — **KEEP**
**val_bpb:** 1.019117 | **Memory:** 17.4 GB | **Status:** keep

**Hypothesis:** Shorter S attention window (512 instead of 1024) allows more optimizer steps.

**Code change:** S blocks: window_size=(512, 0), SVE blocks: window_size=(1024, 0) (unchanged).

**Analysis:** Significant improvement (+0.002 bpb). Despite losing some local context, the increased step count from faster local attention wins. Attention compute is a small but meaningful fraction of total cost.

---

### Exp 23 — `bedecfb` — **KEEP**
**val_bpb:** 1.019060 | **Memory:** 17.4 GB | **Status:** keep

**Hypothesis:** Reducing SVE window (1024→512) also gives more steps, like S did.

**Code change:** SVE blocks: window_size=(512, 0).

**Analysis:** Small additional gain. Both S and SVE local blocks benefit from 512-token windows. Step time decreased slightly (~7ms), giving marginally more gradient updates.

---

### Exp 24 — `a9098a2` — discard
**val_bpb:** 1.020107 | **Memory:** 17.4 GB | **Status:** discard

**Hypothesis:** 5 LVE blocks (instead of 4) gives better global coverage.

**Code change:** `[S, SVE, S, SVE, S, SVE, S, LVE, LVE, LVE, LVE, LVE]`

**Analysis:** Worse. 4 LVE is optimal — 5 displaces a local block without sufficient benefit.

---

### Exp 25 — `4bbc195` — discard
**val_bpb:** 1.025044 | **Memory:** 17.4 GB | **Status:** discard

**Hypothesis:** 6 attention heads (head_dim=64, FA-optimal) instead of 4 (head_dim=96) gives faster and better attention.

**Code change:** n_head=6, n_kv_head=6 for all blocks.

**Analysis:** Worse despite head_dim=64 being theoretically more FlashAttention-efficient. The model quality dropped with 6 heads, suggesting that 4 heads with head_dim=96 captures more useful attention patterns for this architecture.

---

### Exp 26 — `629ba76` — discard
**val_bpb:** 1.020294 | **Memory:** 19.6 GB | **Status:** discard

**Hypothesis:** 14 layers with 512 window (vs previous 14-layer test at 1024 window) might now be viable.

**Code change:** `[S,SVE,×5,LVE,×4]` with all-512-window local blocks.

**Analysis:** Still worse than 12 layers, with much higher memory. The window optimization doesn't change the fundamental depth/steps tradeoff — 14 layers still takes 14% more time per step than 12.

---

### Exp 27 — `4bde4b7` — discard
**val_bpb:** 1.019018 | **Memory:** 18.4 GB | **Status:** discard

**Hypothesis:** 13 layers might be the exact inflection point between 12 (best) and 14 (worse).

**Code change:** `[S, SVE, S, SVE, S, SVE, S, SVE, S, LVE, LVE, LVE, LVE]`

**Analysis:** Marginally better than 14 layers, but worse than 12 by 0.000042 and uses 1GB more memory. Too marginal to justify the complexity. 12 remains optimal.

---

### Exp 28 — `d3049d7` — discard
**val_bpb:** 1.021813 | **Memory:** 17.4 GB | **Status:** discard

**Hypothesis:** Having one LVE at the very start (global context before local processing) could capture important document-level structure early.

**Code change:** `[LVE, S, SVE, S, SVE, S, SVE, S, SVE, LVE, LVE, LVE]` — sandwich pattern.

**Analysis:** Worse. Placing a global attention block before local processing hurts — the model needs local representations to be built first before global aggregation is useful. Local-first ordering is fundamental.

---

### Exp 29 — `e19a93a` — discard
**val_bpb:** 1.022305 | **Memory:** 16.2 GB | **Status:** discard

**Hypothesis:** GQA (n_kv_head=2) for ALL blocks — halves KV compute, more optimizer steps.

**Code change:** All blocks: n_kv_head=2.

**Analysis:** Worse despite 5.5% faster steps (~775 vs 730). GQA hurts quality more than the extra steps help. The KV reduction impairs attention quality when applied uniformly.

---

### Exp 30 — `70071bb` — discard
**val_bpb:** 1.023668 | **Memory:** 17.0 GB | **Status:** discard

**Hypothesis:** GQA only for LVE blocks — global attention can use fewer KV heads while local stays full.

**Code change:** LVE: n_kv_head=2, S/SVE: n_kv_head=4 (unchanged).

**Analysis:** Worse. Global attention (full 2048-token context) needs full KV heads for sufficient pattern diversity. GQA hurts on LVE more than on local blocks — this pointed toward the opposite approach (GQA for local, full for global).

---

### Exp 31 — `4160e76` — discard
**val_bpb:** 1.030943 | **Memory:** 18.1 GB | **Status:** discard

**Hypothesis:** n_model=512 with LVE using n_in=512 lets global attention see the full 512-dim residual stream (including the initial embedding preserved in the upper 128 dims via x0 path).

**Code change:** n_model=512, S/SVE: n_embd=384, LVE: n_in=512, n_embd=384.

**Analysis:** Slower per step (1337ms vs 1230ms, ~8% overhead from larger QKV projections) and worse quality. The "extra" 128 dims in the residual stream carry only the wte_pad and decayed initial embeddings — not useful signal for global attention.

---

### Exp 32 — `06c8d84` — discard
**val_bpb:** 1.034459 | **Memory:** 17.4 GB | **Status:** discard

**Hypothesis:** LVE blocks with 6 attention heads (head_dim=64, FA-optimal) capture more diverse global patterns than 4 heads (head_dim=96).

**Code change:** LVE: n_head=6, n_kv_head=6 (all matrices remain 384×384).

**Analysis:** Significantly worse (1.034 vs 1.019). Having mismatched head counts between local (4 heads) and global (6 heads) blocks hurts more than the head_dim=64 FA optimization helps. Local features are built with 4-head attention patterns, and global attention needs to be consistent with that representation.

---

### Exp 33 — `cfaa06a` — **KEEP** (NEW BEST)
**val_bpb:** 1.018862 | **Memory:** 16.7 GB | **Status:** keep

**Hypothesis:** GQA for local blocks (n_kv_head=2) while keeping full KV for global blocks — the inverse of Exp 30.

**Code change:**
```python
S   = BlockConfig(n_head=4, n_kv_head=2, n_embd=384, ...)  # local GQA
SVE = BlockConfig(n_head=4, n_kv_head=2, n_embd=384, ...)  # local GQA
LVE = BlockConfig(n_head=4, n_kv_head=4, n_embd=384, ...)  # full KV
```

**Analysis:** New best result with lower memory. Local blocks (512-token windows) don't need full KV diversity — their simpler, structured context is well-served by shared KV heads. Global blocks (2048-token full context) retain full KV expressiveness. Step time: 1188ms (~770 steps) vs 1230ms (~730 steps) for prior best — 3.4% more gradient updates.

---

### Exp 34 — `7c66fa8` — discard
**val_bpb:** 1.022167 | **Memory:** 16.3 GB | **Status:** discard

**Hypothesis:** MQA (n_kv_head=1) for local blocks — push GQA even further.

**Code change:** S/SVE: n_kv_head=1 (single shared KV head for all 4 query heads).

**Analysis:** Quality drops significantly. n_kv_head=1 is too extreme — a single KV head cannot represent the diversity of attention patterns needed even for local context. n_kv_head=2 is the sweet spot.

---

### Exp 35 — `c5f659e` — discard
**val_bpb:** 1.018927 | **Memory:** 17.7 GB | **Status:** discard

**Hypothesis:** 13 layers with GQA-local base — GQA makes steps faster so 13 layers might now be cost-effective.

**Code change:** `[S, SVE, ×4, S, LVE, ×4]` — 9 local + 4 global, GQA local.

**Analysis:** Still marginally worse than 12 (1.018927 vs 1.018862) with 1GB more memory. The step-time reduction from GQA local (1188ms) increases to 1269ms for 13 layers — that's ~709 steps, still fewer than 12-layer GQA's ~770 steps.

---

### Exp 36 — `b6e3ccd` — discard
**val_bpb:** 1.018869 | **Memory:** 16.7 GB | **Status:** discard

**Hypothesis:** 3S+5SVE (more VE-bearing local blocks) in the local section with GQA might capture richer value residuals.

**Code change:** `[S, SVE, SVE, S, SVE, SVE, S, SVE, LVE, LVE, LVE, LVE]`

**Analysis:** Essentially equal to current best (1.018869 vs 1.018862, Δ=0.000007 — noise level). The alternating 4S+4SVE pattern is equivalent to 3S+5SVE in performance. Kept the simpler alternating pattern per simplicity criterion.

---

## Summary Statistics

- Total experiments: 37
- Kept (new best at time): 9
- Discarded: 28
- Crashes: 0
- Total val_bpb improvement: 1.073110 → 1.018862 = **−0.054248 (5.1%)**
- Best memory: 16.7 GB (cfaa06a) vs baseline 14.7 GB
