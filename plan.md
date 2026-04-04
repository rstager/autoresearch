# Dynamic Routing Transformer (DRT)

## Architecture Overview

The Dynamic Routing Transformer (DRT) replaces the fixed layer-by-layer execution of a standard transformer with a flat pool of heterogeneous compute primitives — attention heads and feed-forward experts — governed by a learned router that dynamically selects which component contributes to the residual stream at each step. Computation iterates until the router emits a halt signal, allowing adaptive depth per token.

### Core Concept

In a conventional transformer, computation flows rigidly: layer 1 attention → layer 1 FFN → layer 2 attention → layer 2 FFN → ... → layer N FFN. Every token pays the same compute cost regardless of difficulty. An attention head trained at layer 12 can never help at layer 3, even if the same pattern is needed.

DRT dissolves this hierarchy. All attention heads and all FFN/MoE experts from all levels are placed into a single shared pool. A router network examines the current state of the residual stream and selects which component(s) should contribute next. This process repeats until the router decides that further computation would not meaningfully improve the output.

### Component Pool

The pool contains two types of primitives:

**Attention Heads**
- Each head performs multi-head-style attention over the full sequence.
- Heads maintain their own Q/K/V projection weights.
- Different heads specialize in different patterns (local, global, syntactic, semantic, positional, etc.).
- Heads are indexed and individually selectable by the router.

**FFN Experts**
- Each expert is a feed-forward sub-network (typically a 2-layer MLP with activation).
- Experts operate on individual token representations (position-wise).
- Different experts specialize in different transformations of the residual stream.
- In an MoE-derived design, these correspond to the individual experts from each MoE layer.

**Pool Size Example**
- A 32-layer transformer with 16 attention heads per layer and 8 MoE experts per layer yields: 512 attention heads + 256 FFN experts = 768 total components.

### Router Network

The router is a learned network that, at each computation step, takes the current residual stream state and produces:

1. **Component selection distribution**: A probability distribution over all components in the pool (attention heads + FFN experts). The router selects one or more components to fire (top-k selection with k as a hyperparameter, or learned).
2. **Halt probability**: A scalar per token indicating whether computation for that token should stop.

**Router inputs at step t:**
- The current residual stream representation for each token.
- A step embedding (analogous to positional encoding but for computation depth).
- Optionally, a summary of which components have already fired (a binary mask or soft utilization vector).

**Router architecture:**
- A lightweight MLP or cross-attention mechanism over the residual stream.
- Must be small relative to the components it routes to — otherwise the router itself dominates compute cost.
- Produces logits over the component pool, passed through softmax (or Gumbel-softmax during training).

### Computation Step

At each step t:

```
1. Router examines residual stream state x_t
2. Router selects component(s) c_t from pool
3. Selected component(s) compute their output:
   - If c_t is an attention head: output = Attention(x_t; Q_c, K_c, V_c)
   - If c_t is an FFN expert:     output = FFN_c(x_t)
4. Residual update: x_{t+1} = LayerNorm(x_t + output)
5. Router evaluates halt probability h_t for each token
6. If h_t > threshold (or cumulative halt probability exceeds threshold): stop
7. Otherwise: return to step 1
```

### Halting Mechanism

Two options for the halting mechanism:

**Option A — Adaptive Computation Time (ACT) style (differentiable):**
- At each step, the router outputs a halt probability p_t ∈ (0, 1) per token.
- Maintain a running sum R_t = Σ_{i=1}^{t} p_i.
- Halt when R_t ≥ 1 - ε for threshold ε.
- Final output is a weighted combination of intermediate states, weighted by p_t values.
- Advantage: Fully differentiable. Disadvantage: Produces "soft" halting, may waste compute.

**Option B — Discrete halting with RL:**
- Router outputs a binary halt decision at each step.
- Train with REINFORCE or PPO: reward = prediction_quality - β * steps_used.
- Advantage: Crisp halting, explicit compute-quality tradeoff. Disadvantage: Higher variance gradients, harder to train.

**Recommended: Hybrid approach.** Use ACT during pretraining for stable gradients, switch to RL-based discrete halting during fine-tuning.

### Residual Stream Management

Each component's contribution is added to the residual stream with:

- **Per-step LayerNorm**: Applied after each residual update to stabilize the growing representation.
- **Learned scaling**: Each component has a learned scalar gate (initialized near zero) that controls the magnitude of its contribution. This prevents early random components from corrupting the residual stream during training.
- **Step embeddings**: Added to the residual stream at each step so components can condition on "how deep" the computation currently is.

### Differences from Related Architectures

| Feature | Standard Transformer | Universal Transformer | MoE Transformer | DRT |
|---|---|---|---|---|
| Depth | Fixed | Adaptive (same block) | Fixed | Adaptive (any component) |
| Component reuse | No | Yes (single block) | No | Yes (full pool) |
| Sparse activation | No | No | Per-layer | Global |
| Routing | None | None | Per-layer top-k | Global dynamic |
| Component heterogeneity | Per-layer | No (shared) | Per-layer experts | Full heterogeneity |

---

## Training Plan

Training DRT end-to-end from scratch is intractable. The router cannot learn to select useful components when the components themselves output noise. The following staged training plan addresses this.

### Phase 1: Conventional Pretraining (Component Initialization)

**Goal:** Produce well-trained attention heads and FFN experts with known specializations.

**Method:**
1. Train a standard deep transformer (or MoE transformer) on the target pretraining corpus using conventional methods.
2. Architecture: N layers, H attention heads per layer, E experts per layer (if MoE).
3. Train to convergence or near-convergence on standard language modeling objective.
4. Use standard hyperparameters, optimizer (AdamW), learning rate schedule (warmup + cosine decay).

**Outcome:** A fully trained conventional model whose components will seed the DRT pool.

**Hyperparameters (example for a medium-scale prototype):**
- Layers: 12
- Attention heads per layer: 12 (144 total heads)
- MoE experts per layer: 8 (96 total experts)
- Hidden dimension: 768
- Sequence length: 2048
- Training tokens: 100B–300B
- Optimizer: AdamW, lr=3e-4, weight decay=0.1
- Schedule: Linear warmup 2000 steps, cosine decay

### Phase 2: Router Pretraining (Imitation Learning)

**Goal:** Train the router to replicate the original model's fixed layer ordering, providing a strong initialization.

**Method:**
1. Freeze all attention head and FFN expert parameters from Phase 1.
2. Initialize the router network randomly.
3. The "ground truth" routing sequence is the original layer order:
   - Step 1: Layer 1 attention heads
   - Step 2: Layer 1 FFN/experts
   - Step 3: Layer 2 attention heads
   - ...and so on.
4. Train the router to predict this sequence, using the residual stream state at each step as input.

**Loss function:**
```
L_total = L_LM + λ_imit * L_imitation + λ_bal * L_balance + λ_compute * L_compute

Where:
  L_LM        = Standard cross-entropy language modeling loss
  L_imitation  = KL divergence between router's component distribution and
                 the "correct" component at each step (from original layer order)
  L_balance    = Load balancing loss (see below)
  L_compute    = Mean steps per token (ponder cost)
```

**Load balancing loss** (critical for preventing routing collapse):
```
L_balance = N * Σ_i (f_i * p_i)

Where:
  N   = number of components in pool
  f_i = fraction of tokens routed to component i (across batch)
  p_i = mean router probability assigned to component i (across batch)
```
This penalizes correlation between assignment frequency and router confidence, encouraging uniform utilization.

**Hyperparameters:**
- Router architecture: 2-layer MLP, hidden dim = 256
- λ_imit = 1.0 (anneal to 0 over training)
- λ_bal = 0.01
- λ_compute = 0.001
- Optimizer: AdamW, lr=1e-3 for router only
- Training steps: 50K–100K
- Max iterations per token: 2× original layer count (cap for efficiency)

**Outcome:** A router that approximately replicates the original model's behavior, serving as a warm start.

### Phase 3: Joint Fine-tuning (Emergent Routing)

**Goal:** Allow the full system (router + components) to discover novel, potentially superior routing strategies.

**Method:**
1. Unfreeze all parameters (components + router).
2. Use differential learning rates:
   - Components: low lr (1e-5 to 5e-5) — these are already well-trained.
   - Router: moderate lr (1e-4 to 5e-4) — needs freedom to explore.
3. Anneal the imitation loss to zero over the first 10% of this phase, allowing the router to deviate from the original layer ordering.
4. Maintain load balancing and compute cost losses throughout.

**Loss function:**
```
L_total = L_LM + λ_bal * L_balance + λ_compute * L_compute + λ_entropy * L_entropy

Where:
  L_entropy = -Σ_i p_i * log(p_i)  (entropy of router distribution, per step)
```
The entropy term prevents premature collapse of the routing distribution.

**Curriculum for compute budget:**
- Start with a generous max-steps budget (2× original depth).
- Gradually tighten the compute cost penalty λ_compute over training.
- This lets the model first learn what's useful, then learn to be efficient.

**Hyperparameters:**
- Optimizer: AdamW
- Component lr: 5e-5, Router lr: 3e-4
- λ_bal = 0.01
- λ_compute = 0.001 → 0.01 (linear anneal over training)
- λ_entropy = 0.01
- Training steps: 100K–500K (on same or new data)

**Outcome:** A model that routes dynamically, potentially using fewer steps than the original layer count for easy tokens and more for hard tokens.

### Phase 4: Halting Refinement

**Goal:** Sharpen the halting mechanism for efficient inference.

**Method:**
1. Switch from ACT-style soft halting to discrete halting.
2. Use PPO or REINFORCE to train the halt decision:
   - **State:** Current residual stream + step count + router history.
   - **Action:** Halt or continue.
   - **Reward:** R = quality_score - β * step_count.
   - quality_score: negative cross-entropy loss on the next token prediction.
3. Keep component weights and routing weights mostly frozen (or with very low lr).
4. Focus training signal on the halting head of the router.

**Hyperparameters:**
- β (compute penalty): Tune to achieve desired average step count.
  - β = 0.01: Mild pressure, model uses near-maximum steps.
  - β = 0.1: Strong pressure, model aggressively reduces steps.
  - Start with β = 0.01, increase until quality degrades unacceptably.
- RL optimizer: Adam, lr=1e-4
- PPO clip ratio: 0.2
- Training steps: 20K–50K

**Outcome:** A model with crisp, efficient halting behavior.

### Phase 5: Distillation and Deployment Optimization (Optional)

**Goal:** Produce a smaller, faster model that captures the DRT's learned routing patterns.

**Method:**
1. Analyze the trained DRT's routing patterns:
   - What is the average number of steps per token?
   - Which components are used most frequently?
   - Are there clear phase patterns (attention-heavy early, FFN-heavy late)?
2. Prune rarely-used components (those receiving < 1% of traffic).
3. Optionally distill the DRT into a fixed-depth model that mimics its routing patterns, for hardware-efficient deployment.

---

## Diagnostics and Monitoring

### Metrics to Track During Training

**Routing Health:**
- Router entropy per step (should be moderate — not collapsed, not uniform).
- Component utilization histogram (should be roughly uniform with natural variation).
- Average and distribution of steps per token.
- Steps per token vs. token difficulty (measure by loss on that token).

**Quality:**
- Perplexity / cross-entropy loss (standard LM metric).
- Comparison to Phase 1 baseline at equivalent compute.
- Per-token loss stratified by number of steps used.

**Stability:**
- Gradient norms for router vs. components.
- Residual stream norm at each step (should grow smoothly, not explode).
- Halt probability distribution (should be bimodal — confident halt or confident continue).

### Visualization Priorities

1. **Routing heatmap**: For a given input, show which components fire at each step. Compare to the original model's layer ordering.
2. **Step count distribution**: Histogram of steps-per-token across a validation set. Should show meaningful variance, not a spike at one value.
3. **Component specialization**: For each component, show which input patterns trigger it most frequently. Verify that specializations are meaningful.
4. **Phase patterns**: Do attention heads tend to fire before FFN experts? Does the model learn alternating attention/FFN patterns, or something novel?

---

## Known Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Routing collapse | Model uses <10% of components, wasting parameters | Load balancing loss + entropy regularization + monitoring |
| Training instability | Loss spikes, divergence | Per-step LayerNorm, learned scaling gates (init near zero), gradient clipping |
| Credit assignment failure | Router can't learn long routing chains | Staged training (Phase 2 imitation gives short-horizon credit), curriculum on chain length |
| Hardware inefficiency | Poor GPU utilization from variable-length iteration | Pad to max steps within batch during training; bucket sequences by estimated difficulty |
| Ordering degeneracy | Model learns arbitrary orderings that don't generalize | Step embeddings, Phase 2 imitation prior, monitoring for phase patterns |
| Halting too early | Model shortcuts to minimize compute cost | Curriculum on λ_compute (start lenient), quality gates, track loss-vs-steps tradeoff |

---

## Prototype Recommendations

### Minimum Viable Prototype

Start with a small-scale version to validate the concept before investing in full-scale training:

- **Base model**: 4-layer transformer, 4 attention heads per layer, 4 FFN experts per layer.
- **Pool size**: 16 attention heads + 16 FFN experts = 32 components.
- **Hidden dim**: 256.
- **Max steps**: 16 (2× original depth).
- **Task**: Character-level language modeling on a small corpus, or a synthetic task with variable difficulty (e.g., variable-length arithmetic).
- **Validation criterion**: The model should learn to use fewer steps for easy inputs and more steps for hard inputs. If step count is constant across inputs, the adaptive routing is not working.

### Scaling Plan

Once the prototype validates:
1. Scale base model to 12 layers → pool of ~240 components.
2. Scale to 32 layers → pool of ~768 components.
3. At each scale, verify that routing patterns remain non-trivial and quality matches or exceeds the fixed-depth baseline at equivalent average compute.

---

## Summary

The DRT architecture replaces fixed transformer depth with dynamic, router-guided iteration over a shared pool of attention heads and FFN experts. The key insight is that components trained at different depths may be useful at computation stages other than where they were originally placed.

Training requires a careful staged approach:
1. **Pretrain components** conventionally (standard transformer training).
2. **Pretrain router** via imitation of the original layer ordering.
3. **Joint fine-tune** with imitation annealed away, allowing novel routing.
4. **Refine halting** with RL for efficient adaptive computation.

The primary risks are routing collapse and training instability, both addressable with known techniques from the MoE and adaptive computation literature. The primary open question is whether the learned routing strategies will meaningfully outperform fixed layer ordering, or simply rediscover it — answerable only through experiment.
