# Tiny transformer: the from-scratch CPU precursor

Run commands from the repository root with the optional CPU environment. No
model download, pretrained weights, tokenizer download, or network access is
used. This architecture shares the ticket task and prediction format with the
HF runner, **not** its architecture, vocabulary, weights, or adapter format.

```powershell
.\pipeline\.venv\Scripts\python.exe -m pipeline.tiny train --output .\results\pipeline\tiny-fp32
.\pipeline\.venv\Scripts\python.exe -m pipeline.tiny predict --artifact .\results\pipeline\tiny-fp32 --output .\results\pipeline\tiny-predictions.jsonl
.\pipeline\.venv\Scripts\python.exe -m pipeline.tiny quantize --artifact .\results\pipeline\tiny-fp32 --output .\results\pipeline\tiny-int8
.\pipeline\.venv\Scripts\python.exe -m pipeline.tiny predict --artifact .\results\pipeline\tiny-int8 --output .\results\pipeline\tiny-int8-predictions.jsonl
.\pipeline\.venv\Scripts\python.exe -m unittest tests.test_tiny -v
```

Training defaults to `results\pipeline\data`, including `train.jsonl` and
`validation.jsonl`. Prediction uses `test.jsonl` unless `--split` is supplied.
Use `--data-dir` (alias `--data`) to select another generated dataset. Output artifact
directories and prediction files must not already exist.
Each prediction file also has a `.manifest.json` sidecar identifying the model
digest, source manifest, dataset/split, generation settings, and timing scope.

## Defaults and smoke run

The default model has dimension 64, four heads, two pre-normalized residual
blocks, a 4× GELU MLP, and 256 learned absolute positions. Training uses CPU
FP32, two intra-op threads, AdamW, batch size 16, 400 steps, learning rate
0.003, 20 warmup steps, cosine decay to 10% of peak, and gradient clipping at
1.0. Batches are sampled with replacement from a seeded CPU generator.

```powershell
.\pipeline\.venv\Scripts\python.exe -m pipeline.tiny train --output .\results\pipeline\tiny-smoke --dim 32 --layers 1 --batch-size 4 --steps 30 --warmup-steps 3 --eval-every 15 --checkpoint-every 15
```

The smoke configuration checks execution, not useful classification quality.
The initial targeted suite took about three seconds on the development
Windows CPU environment (Python 3.11, torch 2.8.0+cpu); a default 400-step
training run took about 44 seconds excluding process startup. The 30-step smoke
training loop took about 0.5 seconds, also excluding imports/startup. These are
single-machine observations, not speed guarantees. Inspect your own metrics.

## What is actually trained

Vocabulary IDs are PAD=0, BOS=1, ASSISTANT=2, EOS=3, and UTF-8 bytes at IDs
4–259. ASCII task characters therefore each consume one token. Each example is:

```text
[BOS] prompt bytes [ASSISTANT] response bytes [EOS]
```

Inputs and labels are shifted by one position. All prompt and padding targets
are `-100`; only response bytes and EOS enter cross-entropy. The first response
byte is predicted from the ASSISTANT position. Attention uses explicit QKV and
output linear projections plus causal scaled-dot-product attention, not
`MultiheadAttention` internals. Right-side batch padding cannot influence
earlier valid positions. There is no dropout.

Examples too long for the configured context raise an error rather than losing
response targets silently. Increase `--max-seq-len` if necessary. Generation
retains absolute positions, does not slide or truncate context, and recomputes
the full prefix at each autoregressive step; there is **no KV cache**.
Greedy generation stops at EOS, the token limit, or context exhaustion.
It is unconstrained: it does not substitute a classifier or repair invalid JSON.
Decoded output omits special IDs; `generated_tokens` counts every actual model
step, including EOS and any other generated special IDs.

## Artifacts, losses, and exact restart

FP32 directories contain:

* `manifest.json`: architecture, tokenizer, precision, data digest, and progress.
* `model.pt`: inference state dictionary.
* `checkpoint.pt`: latest optimizer/model state, completed step, CPU torch RNG,
  separate batch RNG, full model/train configuration, data digest, and histories.
* `metrics.json`: initial/final validation supervised losses, fixed validation
  IDs, raw per-step training loss/LR/gradient norms, evaluation history, and time.

Validation loss is token-weighted response/EOS cross-entropy over the **same
fixed set** before and after training. The default evaluates all 36 validation
examples; smaller requested samples are spaced across the ordered split.
Loss reduction is not a classification-accuracy result: measure free-running
predictions with the shared evaluator. The synthetic held-out templates are a
limited generalization exercise, not evidence of real support-ticket ability.

Checkpoint writes are atomic replacements every 100 steps and at normal exit.
An abrupt interruption can lose steps since the last checkpoint. Resume into
a new directory and repeat the original configuration, including the **total**
step budget; changing the LR schedule, threads, model, or training/validation
data is rejected. No test-set content enters checkpoint identity or training.

```powershell
# Exercise a deliberate interruption in a 400-step schedule.
.\pipeline\.venv\Scripts\python.exe -m pipeline.tiny train --output .\results\pipeline\tiny-partial --stop-after 100
.\pipeline\.venv\Scripts\python.exe -m pipeline.tiny train --output .\results\pipeline\tiny-resumed --resume .\results\pipeline\tiny-partial
```

`--resume` also accepts the `checkpoint.pt` path directly. `--stop-after` is an
absolute stopping step, not an additional-step count. Exact CPU restart parity
is tested in the same environment; cross-version/hardware bitwise reproducibility
is not promised. All checkpoint loads use `weights_only=True`. Load only your
own artifacts; weights-only loading does not make arbitrary tensor files immune
to excessive allocation or other resource abuse.

## Real CPU INT8, not GPU low-bit serving

Quantization converts **every `nn.Linear`** to torch dynamic quantized CPU
linears. Saved state contains actual qint8 weight codes with scale/zero-point
metadata and biases; loading reconstructs quantized modules and repacks their
weights. It does not load a complete FP32 trained model and call it INT8.
Embeddings, normalization, biases, and attention arithmetic remain floating
point. Activations are dynamically quantized inside the CPU linear kernels.

The INT8 directory contains only `manifest.json` and `model.pt`; it is not a
training-resume checkpoint. Its manifest lists converted layers, quantization
engine, weight schemes, and serialized model sizes. The recorded engine must
be available on the loading CPU build. There is no GPU low-bit kernel, no
embedding quantization, and no guaranteed latency improvement at this size.
Quantization can change greedy tokens and quality. The pinned torch 2.8 API
emits deprecation warnings; migration is needed before upgrading to a release
that removes `torch.ao.quantization`.

## Shared serving interface

```python
from pathlib import Path
from pipeline.tiny import load_generator

generator = load_generator(Path(r"results\pipeline\tiny-int8"), device="cpu")
result = generator.generate("your complete instruction and ticket", max_new_tokens=48)
```

The result has `prediction`, `generated_tokens`, `finish_reason` (`eos`,
`length`, or `context_length`), and `latency_s`. Inference uses eval mode and
`torch.inference_mode`. CPU loading defaults to two process-wide torch
intra-op threads; callers may supply `threads=N`. FP32 inference additionally
accepts CUDA when available, but this lab's training and INT8 path are CPU-only.
There is no batching, streaming, or accelerator training in this module.
