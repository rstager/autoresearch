# run.md — Parallel GPU Task Runner

A reusable protocol for running a list of tasks in parallel across available GPUs. Each task runs in its own isolated git worktree. Results are returned to the caller as a structured report.

This protocol is invoked by `program.md` for each batch of experiments, and can also be used standalone for any ad-hoc parallel workload.

---

## Inputs

The caller provides:

- **`BASE_COMMIT`**: git commit hash that all worktrees will be reset to before starting
- **`TASKS`**: an ordered list of task descriptions — one plain-English string per task (e.g. `"narrow layers 0-1 to n_embd=256"`)
- **`TIMEOUT`** *(optional)*: per-task wall-clock timeout in minutes. Default: 10 minutes.
- **`MODEL`** *(optional)*: sub-agent model override. Default: `opus` (for code changes).

---

## Setup

Do this once at the start of each `run.md` invocation:

1. **Detect GPUs**: `nvidia-smi --query-gpu=index,name --format=csv,noheader`
   Note the GPU indices (e.g. `0`, `1`, `2`, `3`) and total count N.

2. **Compute CPU ranges**: reserve ~10% of CPUs (minimum 2) for system/orchestrator use, divide the remainder evenly across N slots:
   ```bash
   TOTAL_CPUS=$(nproc)
   RESERVED=$(( TOTAL_CPUS / 10 ))
   [ $RESERVED -lt 2 ] && RESERVED=2
   AVAILABLE=$(( TOTAL_CPUS - RESERVED ))
   CPUS_PER_GPU=$(( AVAILABLE / N ))
   # Slot i → cores [i*CPUS_PER_GPU .. (i+1)*CPUS_PER_GPU - 1]
   # e.g. 128 total, 13 reserved → 115 available / 2 GPUs → slot 0: 0-57, slot 1: 58-114
   ```

3. **Create worktrees** (skip any that already exist):
   ```bash
   for i in $(seq 0 $((N-1))); do
     git worktree add /tmp/autoresearch-gpu$i HEAD 2>/dev/null || true
   done
   ```

4. **Reset all worktrees** to `BASE_COMMIT`:
   ```bash
   for i in $(seq 0 $((N-1))); do
     cd /tmp/autoresearch-gpu$i && git reset --hard <BASE_COMMIT>
   done
   ```

5. **Initialise the queue file** at `/tmp/autoresearch-queue.json`:
   ```json
   {
     "tasks": [
       {"id": 0, "description": "<task 0>", "status": "pending", "slot": null},
       {"id": 1, "description": "<task 1>", "status": "pending", "slot": null},
       ...
     ],
     "slots": {
       "0": "free",
       "1": "free",
       ...
     }
   }
   ```
   This file is the shared state. Sub-agents read and write it to claim slots and record completion.

---

## Dispatch loop

Run this loop until all tasks are `done` or `crash`:

1. Read `/tmp/autoresearch-queue.json`.
2. Find the first `pending` task and a `free` slot.
3. If found: **atomically claim the slot** by writing `busy` for that slot and `running` for that task (with the slot index noted). Then spawn a sub-agent (background) — see **Sub-agent Protocol** below.
4. If no free slot is available but tasks remain: wait for a sub-agent to complete (you will be notified), then loop back to step 1.
5. If multiple free slots exist: claim and spawn for each simultaneously before waiting.
6. When all tasks are `done` or `crash`: proceed to **Results**.

> **Lock discipline**: always read the queue file, make your change, and write it back as a single operation before spawning the sub-agent. This prevents two simultaneous dispatches claiming the same slot.

---

## Sub-agent Protocol

Each sub-agent receives:

| Input | Value |
|-------|-------|
| `SLOT` | GPU/worktree index (0, 1, …) |
| `GPU_INDEX` | Physical GPU to use |
| `CPU_RANGE` | CPU core range for this slot (e.g. `0-2`) |
| `WORKTREE` | `/tmp/autoresearch-gpu{SLOT}` |
| `BASE_COMMIT` | Commit to base from |
| `TASK_ID` | Task index in the queue |
| `IDEA` | Task description string |
| `TIMEOUT` | Per-task timeout in minutes |
| `SEED` *(optional)* | Integer seed override (for confirmation runs). When provided, patch `torch.manual_seed` and `torch.cuda.manual_seed` in the script before running, then restore them after. |

**Steps:**

1. `cd <WORKTREE>`
2. `git reset --hard <BASE_COMMIT>` — ensure clean slate
3. Read the current `train.py` to understand the codebase
4. Implement `<IDEA>` by modifying `train.py`
5. `git add train.py && git commit -m "<IDEA>"`
6. Record the new commit hash: `git rev-parse --short HEAD`
7. Run the task, pinned to assigned CPU cores:
   ```bash
   taskset -c <CPU_RANGE> env CUDA_VISIBLE_DEVICES=<GPU_INDEX> uv run train.py > run.log 2>&1
   ```
   Kill and report crash if wall-clock time exceeds `<TIMEOUT>` minutes.
8. Extract results:
   ```bash
   grep "^val_bpb:\|^peak_vram_mb:" run.log
   ```
9. If grep output is empty → crash. Run `tail -n 50 run.log` to diagnose. Attempt a simple fix and re-run if the cause is obvious (typo, missing import). Otherwise treat as crash.
10. **Update the queue file**: set this task's status to `done` (or `crash`), record the commit hash and metrics, and set the slot back to `free`.
11. **Return to orchestrator**:
    ```
    task_id:      <int>
    description:  <IDEA>
    commit:       <7-char hash>
    val_bpb:      <float>
    peak_vram_mb: <float>
    status:       ok | crash
    ```

Sub-agents do **not** make keep/discard decisions. That is the caller's responsibility.

---

## Model assignments

| Role | Model | Reason |
|------|-------|--------|
| Orchestrator (caller) | Opus 4.6 (`claude-opus-4-6`) | Strategic decisions: what to run, what results mean |
| Sub-agent (code + run) | Opus 4.6 (`claude-opus-4-6`) | Code changes require careful reasoning about the codebase |
| Sub-agent (re-run only, no code change) | Sonnet 4.6 (`claude-sonnet-4-6`) | Mechanical: just run the existing commit with a different seed |

Spawn sub-agents with `model="opus"` for tasks that involve modifying code, and `model="sonnet"` for confirmation re-runs (seed sweeps of an already-committed change).

---

## Results

Once all tasks complete, return a summary to the caller:

```
=== run.md results ===
BASE_COMMIT: <hash>

task 0: <description>
  commit:       <hash>
  val_bpb:      <float>
  peak_vram_mb: <float>
  status:       ok | crash

task 1: <description>
  ...
```

The caller interprets results, decides what to keep/discard, updates `results.tsv`, cherry-picks winning commits, and pushes.

---

## Cleanup (optional)

Worktrees at `/tmp/autoresearch-gpu{i}` and `/tmp/autoresearch-queue.json` persist across invocations (worktrees are reused and reset at the start of each run). To fully clean up:

```bash
for i in $(seq 0 $((N-1))); do
  git worktree remove /tmp/autoresearch-gpu$i --force
done
rm -f /tmp/autoresearch-queue.json
```
