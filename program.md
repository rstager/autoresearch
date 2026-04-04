# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Detect GPU count**: Run `nvidia-smi --query-gpu=index,name --format=csv,noheader` to see available GPUs. Note the count N — this determines how many experiments run per batch. GPU/CPU setup and worktree creation are handled by `run.md` when each batch is invoked.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first batch.
7. **Initialize state.md**: Create an initial `state.md` with the session branch, date, hypothesis, and current baseline config. Commit and push it.
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU with a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it as:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> uv run train.py
```

`CUDA_VISIBLE_DEVICES` restricts PyTorch to a single GPU; the script sees it as `cuda:0` regardless of physical index.

**What you CAN change** (the scope of this experiment):

This experiment is specifically about **heterogeneous layer architecture** — varying per-layer width and context access across the depth of the network. The hypothesis is that different layers serve different roles:

- **Early layers** establish local token context; they may need less model width and shorter attention windows.
- **Middle layers** connect and evolve concepts; these benefit most from full width (`n_embd == n_model`) and full context.
- **Late layers** convert internal representations to token predictions; they may again be narrower, but may benefit from a wide `n_in` so they can read the full residual stream from the middle layers.

The parameters you vary:

| Parameter | Where | What it controls |
|-----------|-------|-----------------|
| `n_model` | `GPTConfig` | Width of the full residual stream (shared highway between all layers) |
| `n_embd` | per `BlockConfig` | Width this block reads/writes — its compute budget. Must be ≤ `n_model`. Controls `head_dim = n_embd // n_head`. |
| `n_in` | per `BlockConfig` | Width the attention Q/K/V projections read from. `None` = same as `n_embd`. Set wider (up to `n_model`) to let a narrow layer attend over a richer context slice. |
| `n_head` | per `BlockConfig` | Number of attention heads. Must divide `n_embd`. Adjust when changing `n_embd`. |
| `n_kv_head` | per `BlockConfig` | Number of KV heads (GQA). Must divide `n_head`. |
| `has_ve` | per `BlockConfig` | Whether this block uses value embeddings (ResFormer-style). |
| `window_size` | per `BlockConfig` | `(-1, 0)` = full context; `(k, 0)` = sliding window of size k. |
| `blocks` list | `GPTConfig` | The sequence of `BlockConfig` objects — defines depth and per-layer config. |
| Number of layers | implicit | Change by adding/removing entries in `blocks`. |

**Constraints on these parameters:**
- `n_embd` must divide evenly by `n_head` (head_dim = n_embd // n_head must be a positive integer).
- `n_kv_head` must divide `n_head` and be ≤ `n_head`.
- `n_in` must be ≤ `n_model` (you can't read wider than the residual stream).
- `n_embd` must be ≤ `n_model`.
- The first block's `n_embd` determines the embedding dimension (`wte` outputs `bc0.n_embd`; if `n_model > bc0.n_embd`, the difference is zero-padded via `wte_pad`).
- The last block's `n_embd` determines the `lm_head` input width.

**What you CANNOT change** in this experiment:
- `prepare.py` — read-only. Contains `MAX_SEQ_LEN`, `TIME_BUDGET`, the tokenizer, dataloader, and `evaluate_bpb`.
- The optimizer (`MuonAdamW`), learning rates, batch size, schedules, or any training hyperparameters — keep these fixed so results reflect architecture differences only.
- The model's core components: attention mechanism, MLP structure (`relu²`), normalization (`rms_norm`), rotary embeddings, softcap, value embedding gating. Do not change how the model computes — only how it is configured.
- New packages or dependencies — use only what's in `pyproject.toml`.

**The goal is simple: get the lowest val_bpb.** Find the layer width profile across depth that best fits the hypothesis — narrow early, wide middle, narrow-but-wide-input late — or discover that the data says otherwise.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first batch**: Your very first batch should always be to establish the baseline. Run the training script as-is on all GPUs (same unmodified train.py) and record the baseline val_bpb.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

You can extract the key metrics from the log file:

```
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	val_bpb	memory_gb	status	description	analysis	lessons
```

1. git commit hash (short, 7 chars) — from the worktree where the experiment ran
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, `crash`, `confirm-keep`, or `confirm-discard`
5. short text description of what this experiment tried
6. **analysis**: 1–2 sentences explaining *why* this result succeeded or failed — what mechanism drove the outcome. Be specific: "wider middle layers gave attention heads more representational capacity" not "it worked better".
7. **lessons**: 1 sentence stating the generalizable insight to carry forward — what this tells us about the hypothesis space. E.g. "narrowing early layers below 256 hurts more than it saves" or "n_in wider than n_embd helps late layers even when n_embd is small".

Leave `analysis` and `lessons` blank (empty tab) for `confirm-keep` and `confirm-discard` rows — those inherit from the original run.

Example:

```
commit	val_bpb	memory_gb	status	description	analysis	lessons
a1b2c3d	0.997900	44.0	keep	baseline	Uniform 512-dim 8-layer config	Starting point
b2c3d4e	0.993200	44.2	keep	narrow early: layers 0-1 n_embd=256 (seed=1)	Early layers at half width reduced param count but attention still captured local patterns effectively	Early layers can be narrowed to 256 without hurting quality
b2c3d4e	0.994100	44.2	confirm-keep	narrow early: layers 0-1 n_embd=256 (seed=2)
b2c3d4e	0.993800	44.2	confirm-keep	narrow early: layers 0-1 n_embd=256 (seed=3)
c3d4e5f	1.005000	44.0	discard	narrow late: last 2 layers n_embd=256	Late layers read lm_head input; narrowing lost too much representational capacity at prediction time	Late layers need full width or wide n_in to preserve prediction quality
e5f6g7h	0.000000	0.0	crash	double n_model to 1024	OOM — 1024-dim residual stream exceeds VRAM at batch 128	n_model above 512 requires batch size reduction to fit VRAM
```

## The experiment loop

The orchestrator running this loop should use **Opus 4.6** (`claude-opus-4-6`) — it handles strategic reasoning: interpreting results, generating hypotheses, deciding keep/discard, and synthesizing learnings across batches.

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`). The **main branch** (on the original repo checkout) is the source of truth. Worktrees are temporary workspaces.

LOOP FOREVER:

1. **Review state**: Look at the current branch HEAD commit and `results.tsv`. Note the best val_bpb so far.
2. **Generate N ideas**: Think hard. Come up with N diverse experiment hypotheses (N = number of GPUs). Make them meaningfully different — don't just vary one hyperparameter across all slots. Draw on the `lessons` column of results.tsv and the `state.md` next-directions to guide the choices.
3. **Run the batch**: invoke `run.md` with `BASE_COMMIT=<current HEAD>` and `TASKS=<the N ideas from step 2>`. See [run.md](run.md) for the full parallel execution protocol — GPU/CPU setup, worktree isolation, sub-agent dispatch, and per-task results collection.
4. **Think while waiting**: While the sub-agents run (~5 min), use the time productively. Think hard about the *next* batch of experiments — synthesize lessons learned so far, form hypotheses, draft the next N ideas. Commit these to a provisional next-batch plan (just held in context). Once you've settled on the next batch, stop and wait for the current results.
5. **Collect results**: When all sub-agents complete, collect all results. Identify every run that beats the current best val_bpb (there may be zero, one, or several).
6. **Handle results**:
   - **No improvements**: go to step 7 (log and discard all).
   - **Exactly one improvement**: proceed to confirmation (step 8).
   - **Multiple improvements**: run confirmation on the *best* one only (step 8). Promote the second-best idea to the front of the next batch — it will be tested (from the current HEAD, not the winner) in the next iteration, letting you check whether the two improvements stack.
7. (No improvement path) Mark all as `discard`. Log to results.tsv. Skip to step 11.
8. **Confirmation** (default on, can be disabled): Invoke `run.md` with the **same commit** as `BASE_COMMIT` and 3 tasks of the form `"re-run <IDEA> seed=1"`, `"re-run <IDEA> seed=2"`, `"re-run <IDEA> seed=3"`. Pass `SEED` to each sub-agent so it patches the seed before running without modifying the commit (see `run.md` sub-agent protocol). Use `model="sonnet"` for these re-runs since no code changes are involved.
   - Collect all 3 confirmation val_bpb values. Compute the mean.
   - **If mean val_bpb < current best**: confirmation passed. Cherry-pick the winning commit. Mark original as `keep`, confirmation runs as `confirm-keep`.
   - **If mean val_bpb ≥ current best**: confirmation failed — lucky seed. Mark original as `discard`, confirmation runs as `confirm-discard`. Branch stays at current HEAD.
   - **Re-evaluate next batch**: if confirmation passed, review the provisional next-batch plan (drafted in step 4) in light of the confirmed winner. Does the winner change what's worth trying? Adjust the next batch before proceeding.
9. **Log all results** to `results.tsv` (one row per experiment and per confirmation run, including crashes). Fill in `analysis` and `lessons` for every non-confirmation row.
10. **After any `keep`**: push the branch and update `state.md`:
    ```bash
    git push origin HEAD || true
    ```
    Push failures are non-fatal — log a warning and continue. Then rewrite `state.md` (see **State file** section below). Commit `state.md` and attempt to push it too (`git push origin HEAD || true`). If push fails, the state is still preserved locally and will be pushed on the next successful push.
11. **Reset worktrees**: Each worktree should be reset to the new HEAD before the next batch:
    ```bash
    cd /tmp/autoresearch-gpu$i && git reset --hard <new_HEAD>
    ```
12. **Loop back** to step 1 immediately. Never pause to ask the human.

**Confirmation runs** are enabled by default. To disable (e.g. during rapid early exploration or if GPU time is scarce), the human can say "skip confirmation" at session start. When disabled, step 6 reduces to: cherry-pick immediately on any improvement, mark as `keep`.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), the sub-agent should fix it and re-run. If the idea itself is fundamentally broken, just report crash and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes and you have 4 GPUs, you can run approx 48/hour, for a total of about 400 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## State file

`state.md` is a human-readable summary of session progress, committed and pushed to the branch after every `keep`. It allows the experiment to be resumed after an interruption without losing context.

**Location**: `state.md` in the repo root (tracked by git, pushed to remote).

**Update it after every confirmed `keep`** by rewriting the entire file. It should contain:

```markdown
# Experiment State

## Session
- Branch: autoresearch/<tag>
- Started: <date>
- Last updated: <date/time>
- Best val_bpb so far: <float> (commit <hash>)

## Hypothesis being tested
<One paragraph reminding the orchestrator of the architectural hypothesis:
narrow early layers, wide middle, narrow-but-wide-input late layers.
Update this if the focus has shifted based on what's been learned.>

## What has been tried
<Bullet list of experiments, grouped by theme. Include what worked, what
didn't, and any patterns noticed. This is the orchestrator's memory — write
it so a fresh Claude instance can pick up where you left off without reading
every results.tsv row.>

## Current best config
<Paste the relevant BlockConfig / GPTConfig lines from the current train.py
so the baseline is immediately visible on resume.>

## Next directions to explore
<2–4 concrete hypotheses worth trying next, based on what's been learned.
Update this after each batch so it's always forward-looking.>
```

**On resume**: Read `state.md` first. It tells you where you are, what's been learned, and what to try next. Then read `results.tsv` for the full numerical record. Then read the current `train.py` for the exact config.

## Parallel execution

All parallel task dispatch, GPU/CPU isolation, worktree management, and sub-agent protocol is handled by [run.md](run.md). Refer to that document when invoking a batch of experiments or confirmation runs.
