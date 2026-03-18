# CORE Evaluation

`eval_core.py` is a standalone evaluator for the [CORE benchmark](https://arxiv.org/abs/2406.11794) (ICL accuracy) and BPB (bits-per-byte). It has no dependency on the nanochat package and can be dropped into any repository.

The eval bundle is auto-downloaded on first run to `~/.cache/nanochat/eval_bundle/`.

## Programmatic usage (recommended for autoresearch)

```python
import torch
from eval_core import evaluate_core, evaluate_bpb, ModelAdapter, build_token_bytes

device = torch.device("cuda")

# ModelAdapter bridges the autoresearch forward() signature:
#   model(idx, targets, reduction=...) -> loss
# to the interface eval_core expects:
#   model(input_ids, targets, loss_reduction=...) -> loss
adapter = ModelAdapter(model, loss_reduction_kwarg='reduction')

# CORE evaluation
results = evaluate_core(adapter, tokenizer, device, max_per_task=-1)
print(f"CORE metric: {results['core_metric']:.4f}")
# results['results']          -> {task_label: accuracy}
# results['centered_results'] -> {task_label: centered_accuracy}

# BPB evaluation (bring your own dataloader)
token_bytes = build_token_bytes(tokenizer, device=device)
bpb = evaluate_bpb(adapter, your_dataloader, steps=100, token_bytes=token_bytes)
print(f"BPB: {bpb:.6f}")
```

## CLI usage

**HuggingFace model:**
```bash
uv run python eval_core.py --hf-path openai-community/gpt2 --eval core --max-per-task 100
```

**Custom factory function** (a callable in your module that returns `(model, tokenizer)`):
```bash
uv run python eval_core.py --model-factory "mymodule:load_model" --eval core
```

**Write results to a CSV:**
```bash
uv run python eval_core.py --hf-path openai-community/gpt2 --eval core --output results.csv
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-path` | — | HuggingFace model path (mutually exclusive with `--model-factory`) |
| `--model-factory` | — | `module:callable` returning `(model, tokenizer)` |
| `--eval` | `core` | Comma-separated modes: `core`, `bpb` |
| `--max-per-task` | `-1` (all) | Cap examples per task for quick/approximate evals |
| `--output` | `core_eval_results.csv` | CSV output path for CORE results |
| `--device-type` | autodetect | `cuda`, `cpu`, or `mps` |
| `--device-batch-size` | `32` | Per-device batch size (BPB only) |
| `--split-tokens` | `20971520` | Tokens per split (BPB only) |

## ModelAdapter

`ModelAdapter` wraps any `nn.Module` to match the expected interface:

```python
ModelAdapter(model, loss_reduction_kwarg='loss_reduction', max_seq_len=None)
```

- `loss_reduction_kwarg`: the name of the reduction argument in your model's `forward()`.
  Use `'reduction'` for autoresearch, `'loss_reduction'` for nanochat (default).
- `max_seq_len`: if set, sequences longer than this are truncated during CORE eval.

## Public API

| Symbol | Description |
|--------|-------------|
| `evaluate_core(model, tokenizer, device, max_per_task=-1)` | Run full CORE benchmark, return results dict |
| `evaluate_bpb(model, batches, steps, token_bytes)` | Compute BPB over a dataloader |
| `build_token_bytes(tokenizer, device)` | Build vocab byte-length tensor for BPB |
| `ModelAdapter(model, ...)` | Adapter wrapping any nn.Module |
| `write_core_csv(core_results, output_path)` | Write results dict to CSV |
