# Real causal-LM adapter lab

`python -m pipeline.hf` runs actual Transformers/PEFT training, deterministic
generation, and high-precision adapter merging. Run from the repository root,
using the explicitly installed environment described in `ENVIRONMENTS.md`.
There are **no automatic installs, downloads, remote Python execution, guessed
chat templates, or silent CPU fallbacks**.

The suggested base is `Qwen/Qwen2.5-0.5B-Instruct`. Supply either an existing local
HF snapshot directory (`--base C:\Models\Qwen2.5-0.5B-Instruct`) or a Hub
`owner/repository` ID **and an immutable 40-character `--revision` commit SHA**.
Hub access also requires `--allow-download`; otherwise all loads use
`local_files_only=True`. A revision is required even for cached Hub snapshots.
Every command accepts `--cache-dir`, defaulting to `results\pipeline\hub` under
the repository working directory. Both model and tokenizer Hub loads use this
explicit cache, not hidden global profile storage. Choose a different cache path
explicitly when reusing an existing snapshot cache; changing the cache location
does not change the pinned base identity or resume compatibility.
No commands below download a model.

## Prepare and inspect

```powershell
# Only if the shared dataset has not already been generated:
python -m pipeline.data --output results\pipeline\data
$base = 'C:\Models\Qwen2.5-0.5B-Instruct'
python -m pipeline.hf inspect --base $base --split train --limit 3
```

The data contract is `pipeline.data.load_split(data_dir, split)`. Every prompt
already includes the task instruction and ticket. Training loads train and
validation only; prediction defaults to test. `--data-dir` defaults to
`results\pipeline\data`; `--data` is an alias.

`inspect` reports full token counts, `[start, end)` assistant-supervision spans,
IDs, decoded supervised text, and examples exceeding `--max-seq-length` (512).
The tokenizer's chat template formats a user message and the canonical JSON
assistant response. Labels come from offset mappings of the **full conversation**,
not the length of a separately tokenized prefix. A token crossing the textual
assistant boundary is included in the inspected assistant span because it contains
response text. Such tokens are explicitly flagged by inspection. Before training,
the runner also requires that the tokenized generation prefix equals the tokens
before that span. Boundary merges that change this context are rejected rather
than silently teaching a different conditioning sequence. Templates whose
generation prefix is not a textual prefix of the full conversation are rejected. EOS is supervised; if
the template does not supply it in the assistant span, it is appended. Padding
is masked by position even when its ID equals EOS. Training never silently
truncates either prompt or response: oversized examples fail before model loading.
Fast tokenizers with an EOS token and a chat template are required.

## LoRA and NF4 QLoRA

```powershell
python -m pipeline.hf train --base $base --device cuda --method lora `
  --output results\pipeline\hf-lora --max-steps 100
python -m pipeline.hf train --base $base --device cuda --method qlora `
  --output results\pipeline\hf-qlora --max-steps 100
```

Defaults: microbatch 1, accumulation 4, sequence limit 512, rank 8, alpha 16,
dropout 0.05, all-linear LoRA targets, gradient checkpointing, AdamW,
learning rate 0.0002, weight decay 0.01, five warmup updates, then linear decay,
gradient norm clipping at 1.0. CUDA defaults to bfloat16; CPU requires float32.
Use `--dtype float32` explicitly on CUDA hardware without bfloat16 support.
`--device cpu --method lora` is supported for correctness exercises, not a
promise of useful pretrained-model training speed.

QLoRA needs the separately installed bitsandbytes requirements and CUDA. It uses
NF4, double quantization, and the selected compute dtype. All base parameters
are frozen; only LoRA parameters may train. GPU selection uses the current CUDA
device; set `CUDA_VISIBLE_DEVICES` before starting Python if needed. This is
single-process, single-device training, not DDP/FSDP. There is no CPU offloading.
A 6 GiB GPU is a constrained target, **not an assurance that every configuration
fits**. Start with the 0.5B base, microbatch 1, and 512 tokens; reduce sequence
length only after inspecting examples, or use NF4. OOMs remain visible errors.

`--max-steps` counts optimizer updates, not microbatches. Every update consumes
exactly `microbatch * gradient_accumulation` records from a deterministically
shuffled stream, wrapping into the next epoch if necessary. Accumulation averages
microbatch token-mean losses. Validation is a supervised-token-weighted mean.
Losses and gradients must be finite before any update.

Each **new** output directory contains:

- `run.json`: base/data/version/hyperparameter identity.
- `metrics.jsonl`: update loss, learning rate, pre-clip gradient norm, stream
  cursor, supervised tokens, synchronized update timing, and periodic validation.
- `checkpoint-00000010\` etc.: adapter, tokenizer, manifest, safe-loadable
  optimizer/scheduler/RNG/stream state, and a last-written completion marker.
- `adapter\`: final adapter, tokenizer, and source manifest.
- `summary.json`: trainable/total parameter counts, timing, peak CUDA allocated
  and reserved bytes, and completion status. CPU memory fields are null, not
  fabricated GPU measurements. Quantized parameter storage counts are not a
  model-quality or memory-equivalence claim.

Checkpointing and validation default to every 10 updates and always happen at
the requested stop. Timing excludes initial model loading and final adapter
export; training wall time includes validation and intermediate checkpoint I/O.
CUDA is synchronized at measurement boundaries.

## Exact-stream resume

```powershell
python -m pipeline.hf train --base $base --device cuda --method lora `
  --output results\pipeline\hf-part --max-steps 100 --stop-after-steps 20
python -m pipeline.hf train --base $base --device cuda --method lora `
  --output results\pipeline\hf-resumed --max-steps 100 `
  --resume results\pipeline\hf-part\checkpoint-00000020
```

Resume always writes a **new output directory**, never mutates the old run, and
continues at step 21 with the original 100-step scheduler. Repeat any nondefault
training flags. Base identity, complete train/validation contents, package
versions, device/dtype, and training hyperparameters must match. A completed
schedule cannot be extended via resume: that would change the LR schedule.
`--stop-after-steps`, reporting cadence, download permission, and output path do
not affect the schedule identity. Local base files are SHA-256 fingerprinted
(including weights); hashing large snapshots takes time.

State contains optimizer moments, scheduler position, shuffle epoch/cursor,
Python/NumPy/PyTorch/CUDA RNGs; saves occur at optimizer boundaries, never with
pending accumulated gradients. Loading uses `torch.load(weights_only=True)`
with no unsafe pickle fallback. Incomplete checkpoints are rejected. A failed
run can leave a partial directory: preserve it for diagnosis and choose another
new output path. Exact CPU resume is tested with dropout enabled. GPU kernel
nondeterminism may prevent bitwise identity even with the same sample/RNG stream.

## Base, adapter, and runtime-quantized prediction

```powershell
python -m pipeline.hf predict --base $base --device cuda `
  --output results\pipeline\base-predictions.jsonl
python -m pipeline.hf predict --base $base --device cuda `
  --adapter results\pipeline\hf-lora\adapter `
  --output results\pipeline\adapter-predictions.jsonl
python -m pipeline.hf predict --base $base --device cuda --load-in-4bit `
  --adapter results\pipeline\hf-lora\adapter `
  --output results\pipeline\nf4-runtime-predictions.jsonl
```

Generation uses the same user chat prefix, greedy `do_sample=False`, one beam,
and 32 maximum new tokens. Set `--split`, `--limit`, or `--max-new-tokens` as
needed. The prompt plus generation budget must fit `--max-seq-length` and the
model context limit. Input tokens are sliced off before decoding. Each exclusive
JSONL file contains:

```json
{"id":"test-example","prediction":"{\"category\":\"billing\"}","latency_s":0.42,"generated_tokens":7,"completed":true,"finish_reason":"eos"}
```

A neighboring `.manifest.json` sidecar records base/adapter identities, the full
dataset digest, selected split, generation settings, package versions, and
measurement scope. The scorer rejects a sidecar belonging to another dataset
or split. New result files and sidecars must both use unused paths.

All three splits are supported. For `pipeline.distill`, generate teacher
predictions for **training IDs**, without `--limit`:

```powershell
python -m pipeline.hf predict --base $base --device cuda --split train `
  --output results\pipeline\teacher-train-predictions.jsonl
```

Add `--adapter` if the teacher is an adapter model. The prediction runner never
changes dataset labels; the distillation dataset builder owns that operation
and preserves held-out gold labels.

`completed` means an EOS stop, **not** valid JSON or correct classification.
`finish_reason: "length"` flags a token-budget stop. Measure correctness with the
separate evaluator. Timings cover synchronized `generate`, excluding tokenization
and decoding; the first request is cold, with no hidden warmup. This is a
sequential reference runner, not a production throughput benchmark. Adapters
must be lab-produced adapters/checkpoints with matching source manifests.

## Standalone high-precision export

```powershell
python -m pipeline.hf export --base $base --device cpu --dtype float32 `
  --adapter results\pipeline\hf-qlora\adapter `
  --output results\pipeline\merged-hf
python -m pipeline.hf predict --base results\pipeline\merged-hf --device cpu `
  --output results\pipeline\merged-predictions.jsonl
```

Export reloads the original **nonquantized** base and safely merges the adapter,
then saves standard HF model weights/config/generation config plus tokenizer,
source identity, and completion marker. It does not merge into NF4 weights,
download an alternative model, or convert to GGUF. CPU float32 requires enough
RAM for the base and merge; CUDA bfloat16 is also supported. Full-precision export
from QLoRA need not exactly reproduce NF4-runtime outputs due to quantization
differences. Keep the adapter and source identity for reproducibility.

## Local serving integration

The shared server can load the merged export through this interface:

```python
from pathlib import Path
from pipeline.hf import load_generator

generator = load_generator(Path(r"results\pipeline\merged-hf"), device="cpu")
result = generator.generate("Route this support ticket: I was charged twice.", max_new_tokens=48)
```

`load_generator(artifact: Path, device="cpu")` accepts only a completed local
`pipeline.hf export` directory, not a Hub ID or adapter. It returns an object
with `.generate(prompt: str, max_new_tokens: int = 48) -> dict`, containing
`prediction`, `generated_tokens`, `finish_reason`, `latency_s`, and `completed`.
Both model and tokenizer loads are strictly local, with remote code disabled.
The helper shares the CLI's greedy generation, chat formatting, completion
handling, and synchronized timing implementation.

Supported devices are explicitly `cpu` (float32) and `cuda` (bfloat16, requiring
hardware support); no fallback or runtime quantization occurs. The context
budget is the smaller of 512 tokens and the model context limit, including
requested output tokens. Invalid or oversized requests raise `ValueError`,
never truncate. Load once at server startup and reuse the generator. The caller
owns HTTP handling, concurrency/admission control, and request limits.

## Offline validation and limits

```powershell
python -m unittest tests.test_hf -v
```

Tests construct a tiny randomly initialized GPT-2 and local character tokenizer,
train adapters on synthetic records, verify exact interrupted/resumed weights
and metrics, enforce EOS/pad/boundary masks, generate, merge, compare logits, and
reload the merged directory without Hub access. They write uniquely named
scratch directories under `results` and remove them afterward. Heavy tests
explicitly skip if optional dependencies are absent. These tests are **not
evidence of a tested pretrained Qwen GPU run, successful RTX 4050 QLoRA job, or
accuracy improvements**.
