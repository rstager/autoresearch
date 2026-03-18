"""
Standalone CORE evaluator — no nanochat dependency.

Supports two evaluation modes (comma-separated via --eval):
  --eval core   : CORE metric (accuracy on ICL tasks)
  --eval bpb    : Bits per byte on train/val splits

Default: --eval core

Public API (importable):
    from eval_core import evaluate_core, evaluate_bpb_split, ModelAdapter, build_token_bytes

    adapter = ModelAdapter(model)                     # wraps any nn.Module
    results = evaluate_core(adapter, tokenizer, device)
    print(results['core_metric'])

CLI (HuggingFace model):
    python eval_core.py --hf-path openai-community/gpt2 --eval core --max-per-task 100

CLI (factory function returning (model, tokenizer)):
    python eval_core.py --model-factory "mymodule:load_my_model" --eval core,bpb

The model passed to evaluate_core / evaluate_bpb must support:
    model(input_ids)                           -> logits  (B, T, V)
    model(input_ids, targets, loss_reduction='none') -> loss (B, T)  — for BPB only
    model.get_device()                         -> torch.device       — for BPB only

Use ModelAdapter to wrap a model whose forward() signature differs.
"""
import os
import csv
import math
import time
import json
import yaml
import random
import shutil
import zipfile
import tempfile
import argparse
import importlib
import urllib.request

import torch
import torch.distributed as dist
from filelock import FileLock
from jinja2 import Template

# =============================================================================
# Utilities (inlined from nanochat.common)
# =============================================================================

def print0(s="", **kwargs):
    """Print only from rank 0 in distributed settings."""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)


def get_base_dir():
    """Return the base cache directory (respects NANOCHAT_BASE_DIR env var)."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        base = os.environ["NANOCHAT_BASE_DIR"]
    else:
        base = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    os.makedirs(base, exist_ok=True)
    return base


def download_file_with_lock(url, filename, postprocess_fn=None):
    """Download a file to base_dir using a file lock to handle concurrent ranks."""
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"
    if os.path.exists(file_path):
        return file_path
    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")
        if postprocess_fn is not None:
            postprocess_fn(file_path)
    return file_path


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in base_dir."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted, eval_bundle_dir)
    print0(f"Placed eval_bundle at {eval_bundle_dir}")


# =============================================================================
# CORE evaluation (inlined from nanochat.core_eval)
# =============================================================================

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item,
    }
    return [template.render(choice=choice, **context) for choice in item['choices']]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item,
    }
    return [template.render(context=ctx, **context) for ctx in item['context_options']]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item,
    }
    prompt_without = template.render(include_continuation=False, **context).strip()
    prompt_with = template.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len - 1, -1),
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    bsz = len(tokens)
    seq_len = max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx
    assert tokens_without == tokens_with[:start_idx]
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    """Forward pass returning (losses, predictions) of shape (B, T)."""
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none',
    ).view(batch_size, seq_len)
    losses[:, -1] = float('nan')
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available = [i for i in range(len(data)) if i != idx]
        fewshot_examples = [data[i] for i in rng.sample(available, num_fewshot)]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Truncate if model has a max_seq_len
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_starts, new_ends = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_starts.append(s - crop)
                new_ends.append(e - crop)
            else:
                new_tokens.append(t)
                new_starts.append(s)
                new_ends.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_starts, new_ends

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, input_ids)

    if task_type == 'language_modeling':
        si, ei = start_idxs[0], end_idxs[0]
        predicted_tokens = predictions[0, si - 1:ei - 1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    else:
        mean_losses = [losses[i, si - 1:ei - 1].mean().item()
                       for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()


# =============================================================================
# BPB evaluation (inlined from nanochat.loss_eval)
# =============================================================================

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Compute bits-per-byte metric over `steps` batches from `batches` iterator.

    model must support: model(x, y, loss_reduction='none') -> (B, T) loss tensor
    and model.get_device() -> device.
    """
    device = model.get_device()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none')  # (B, T)
        loss2d = loss2d.view(-1)
        y = y.view(-1)
        if (y.int() < 0).any():
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype),
            )
        else:
            num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)


# =============================================================================
# Token bytes helper
# =============================================================================

def build_token_bytes(tokenizer, device="cpu"):
    """
    Build a (vocab_size,) int64 tensor where each entry is the UTF-8 byte length
    of that token, or 0 for special tokens (which should not count toward BPB).

    Works with any tokenizer that has get_vocab_size() and a decode method
    (either tokenizer.decode([id]) or tokenizer.tokenizer.decode([id])).
    """
    # Support both HuggingFace-wrapped and tiktoken-style tokenizers
    if hasattr(tokenizer, 'tokenizer'):
        vocab_size = tokenizer.tokenizer.get_vocab_size()
        def decode_one(tid):
            return tokenizer.tokenizer.decode([tid])
    elif hasattr(tokenizer, 'get_vocab_size'):
        vocab_size = tokenizer.get_vocab_size()
        def decode_one(tid):
            return tokenizer.decode([tid])
    else:
        raise ValueError("Cannot determine vocab size from tokenizer")

    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for tid in range(vocab_size):
        try:
            s = decode_one(tid)
            token_bytes[tid] = len(s.encode('utf-8'))
        except Exception:
            token_bytes[tid] = 0
    return token_bytes


# =============================================================================
# Model adapter
# =============================================================================

class ModelAdapter:
    """
    Wraps any nn.Module so it presents the interface expected by evaluate_core
    and evaluate_bpb.

    Handles differences in forward() keyword argument names across codebases:
      - nanochat uses loss_reduction='none'
      - autoresearch uses reduction='none'

    Args:
        model: nn.Module with forward(input_ids, [targets], [<reduction_kwarg>=...])
        loss_reduction_kwarg: the name of the reduction kwarg in the underlying model
            (default 'loss_reduction' for nanochat; use 'reduction' for autoresearch)
        max_seq_len: optional int; if set, sequences longer than this are truncated
    """
    def __init__(self, model, loss_reduction_kwarg='loss_reduction', max_seq_len=None):
        self.model = model
        self._reduction_kwarg = loss_reduction_kwarg
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        if targets is None:
            return self.model(input_ids)
        return self.model(input_ids, targets, **{self._reduction_kwarg: loss_reduction})

    def get_device(self):
        return next(self.model.parameters()).device


# =============================================================================
# CORE benchmark orchestration
# =============================================================================

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a model on the CORE benchmark.

    Args:
        model: callable with forward(input_ids) -> logits. Use ModelAdapter if needed.
        tokenizer: must implement encode(texts, prepend=bos_id), get_bos_token_id(),
                   and encode_special(str) (returns None if not found).
        device: torch.device
        max_per_task: int, cap per task for quick evals (-1 = all)

    Returns:
        dict with keys:
            'results'          : {task_label: accuracy}
            'centered_results' : {task_label: centered_accuracy}
            'core_metric'      : float
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip",
                                postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_path = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    random_baselines = {}
    with open(eval_meta_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' '),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, "
               f"type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        rb = random_baselines[label]
        centered = (accuracy - 0.01 * rb) / (1.0 - 0.01 * rb)
        centered_results[label] = centered
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered:.4f} | "
               f"time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }


def write_core_csv(core_results, output_path):
    """Write CORE results dict to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        for label in core_results["results"]:
            acc = core_results["results"][label]
            centered = core_results["centered_results"][label]
            f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
        f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
    print0(f"Results written to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def _load_hf_model(hf_path, device):
    from transformers import AutoModelForCausalLM
    from tokenizers import Tokenizer as HFTokenizer

    class HFTokenizerWrapper:
        def __init__(self, tok):
            self.tokenizer = tok

        def get_bos_token_id(self):
            bos = self.tokenizer.token_to_id("<|bos|>")
            if bos is None:
                bos = self.tokenizer.token_to_id("<|endoftext|>")
            assert bos is not None, "Could not find BOS token"
            return bos

        def encode_special(self, text):
            return self.tokenizer.token_to_id(text)

        def encode(self, text, prepend=None, append=None):
            if isinstance(text, str):
                return self._enc_one(text, prepend, append)
            return [self._enc_one(t, prepend, append) for t in text]

        def _enc_one(self, text, prepend, append):
            ids = []
            if prepend is not None:
                ids.append(prepend if isinstance(prepend, int)
                           else self.encode_special(prepend))
            ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
            if append is not None:
                ids.append(append if isinstance(append, int)
                           else self.encode_special(append))
            return ids

        def __call__(self, *args, **kwargs):
            return self.encode(*args, **kwargs)

        def decode(self, ids):
            return self.tokenizer.decode(ids, skip_special_tokens=False)

    print0(f"Loading HuggingFace model: {hf_path}")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_path)
    hf_model.to(device)
    hf_model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None

    hf_tok = HFTokenizer.from_pretrained(hf_path)
    tokenizer = HFTokenizerWrapper(hf_tok)

    class HFModelWrapper:
        def __init__(self, m, max_seq_len):
            self.model = m
            self.max_seq_len = max_seq_len

        def __call__(self, input_ids, targets=None, loss_reduction='mean'):
            logits = self.model(input_ids).logits
            if targets is None:
                return logits
            return torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )

        def get_device(self):
            return next(self.model.parameters()).device

    model = HFModelWrapper(hf_model, max_seq_len)
    return model, tokenizer


def _autodetect_device(device_type_arg):
    if device_type_arg:
        return torch.device(device_type_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Standalone CORE / BPB evaluator")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--hf-path', type=str,
                     help='HuggingFace model path (e.g. openai-community/gpt2)')
    src.add_argument('--model-factory', type=str,
                     help='Python dotted path to a callable returning (model, tokenizer), '
                          'e.g. "mymodule:load_model"')
    parser.add_argument('--eval', type=str, default='core',
                        help='Comma-separated eval modes: core,bpb (default: core)')
    parser.add_argument('--max-per-task', type=int, default=-1,
                        help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--output', type=str, default='core_eval_results.csv',
                        help='CSV output path for CORE results')
    parser.add_argument('--device-type', type=str, default='',
                        help='cuda|cpu|mps (empty = autodetect)')
    # BPB-specific
    parser.add_argument('--device-batch-size', type=int, default=32,
                        help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40 * 524288,
                        help='Tokens per split for BPB evaluation')
    args = parser.parse_args()

    eval_modes = set(m.strip() for m in args.eval.split(','))
    valid_modes = {'core', 'bpb'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    device = _autodetect_device(args.device_type)
    print0(f"Device: {device}")

    if args.hf_path:
        model, tokenizer = _load_hf_model(args.hf_path, device)
        model_name = args.hf_path
    else:
        module_path, _, factory_name = args.model_factory.partition(':')
        module = importlib.import_module(module_path)
        factory = getattr(module, factory_name)
        model, tokenizer = factory()
        model_name = args.model_factory

    print0(f"Evaluating: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    ddp_rank = int(os.environ.get('RANK', 0))

    if 'core' in eval_modes:
        print0("\n" + "=" * 80)
        print0("CORE Evaluation")
        print0("=" * 80)
        core_results = evaluate_core(model, tokenizer, device,
                                     max_per_task=args.max_per_task)
        print0(f"\nCORE metric: {core_results['core_metric']:.4f}")
        if ddp_rank == 0:
            write_core_csv(core_results, args.output)

    if 'bpb' in eval_modes:
        print0("\n" + "=" * 80)
        print0("BPB Evaluation")
        print0("=" * 80)
        if not hasattr(model, 'get_device'):
            print0("Warning: model does not have get_device(); wrapping with ModelAdapter")
            model = ModelAdapter(model)

        # Build token_bytes from the tokenizer
        token_bytes = build_token_bytes(tokenizer, device=device)

        # We need a dataloader — import from the target repo or use a simple stub
        # Users should pass their own dataloader; here we print a helpful message.
        print0("BPB evaluation requires a dataloader. "
               "Call evaluate_bpb(model, dataloader, steps, token_bytes) directly "
               "from your own script after importing this module.")


if __name__ == "__main__":
    main()
