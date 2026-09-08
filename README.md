# LLM Infrastructure: Models, GPUs, Training, and Serving

This material connects the model's computation to the system that trains and
serves it. It starts with the mathematics of prediction and learning, then
covers GPU execution, fine-tuning, distillation, numerical compression,
request-state management, and serving under load.

The running question is: **what work is being done, what data must be retained,
and how do we establish whether a change helped?**

## Learning path

Start with [the foundations](foundations/README.md) if embeddings, gradients,
or transformer blocks are unfamiliar. Then work through [the infrastructure
course](course/README.md).

| Part | Contents |
| --- | --- |
| [Model foundations](foundations/README.md) | Linear algebra, tokens, embeddings, logits, loss, backpropagation, transformer architecture, and generation |
| [GPU systems](course/01-gpu-systems.md) | Execution and memory hierarchy, arithmetic intensity, kernels, precision, timing, and profiling |
| [Training and fine-tuning](course/02-training-and-fine-tuning.md) | Data/label construction, SFT, training memory, accumulation, checkpointing, LoRA, QLoRA, and export |
| [Distillation](course/03-distillation.md) | Teacher signals, soft targets, temperature-scaled objectives, sequence distillation, and evaluation |
| [Weight and activation quantization](course/04-weight-and-activation-quantization.md) | Scales, grouping, outliers, calibration, PTQ/QAT, precision, and kernel compatibility |
| [KV cache and its quantization](course/05-kv-cache-and-quantization.md) | State lifetime, architecture-specific accounting, paging, prefix reuse, precision, and quality |
| [Serving and distributed systems](course/06-serving-and-distributed-systems.md) | Batching, scheduling, latency/throughput, distributed parallelism, and operational tradeoffs |

The [foundations exercises](foundations/EXERCISES.md) include worked answers.
The [infrastructure workbook](course/WORKBOOK.md) adds practical labs and four
capstones: single-GPU inference, GPU adapter training, distillation plus
quantization, and serving capacity under load.

Diagrams are embedded in the chapters. The [visual index](assets/README.md)
links to four step-by-step animations and the structural diagrams. Animations
are collapsed by default; static images show the same ideas without motion.

## Run the small experiments

Use Python 3.10 or newer. Install NumPy if it is missing:

```powershell
python -m pip install -r requirements.txt
```

From the repository root:

```powershell
python foundations\01_tokens_and_probabilities.py
python foundations\02_train_a_small_model.py
python foundations\03_transformer_block.py

python -m course.demos.lora
python -m course.demos.distillation
python -m course.demos.quantization
```

These CPU experiments expose the actual operations: explicit gradient updates,
a frozen base with a learned adapter, soft-target distillation, and code/scale
storage. No model downloads are needed.

The GPU fine-tuning and distributed capstones require a separately selected
training/serving stack and suitable hardware. The course supplies the procedure,
calculations, and evaluation requirements, not a preconfigured GPU SFT runner.

## Inference examples and case study

```powershell
python 01_why_kv_cache.py
python 02_kv_cache_size.py --context 8192 --sequences 2
python 03_kv_quant_error.py
python 04_gqa_explained.py
python 05_flash_attention.py
python 06_read_inference_results.py
python 07_performance_budget.py
```

[The laptop investigation](GUIDE.md) analyzes a Qwen3.5-4B deployment on a
6 GB RTX 4050. The first request took about 102 seconds to start answering;
the warm repeat started in 0.19 seconds. The record helps distinguish loading,
prefill, and decode rather than treating that one run as a hardware benchmark.

[SETUP.md](SETUP.md) explains how to reuse an existing Ollama installation.
[The inference labs](LABS.md) include a streaming client and generated retrieval
documents for controlled input-length experiments.

```powershell
python 08_local_client.py --prompt "Explain prefill versus decode." --save results\first.json
```

The client defaults to the original deployment on `http://127.0.0.1:11435`.
Use `--url http://127.0.0.1:11434` for the usual Ollama port.

## Repository boundaries

Weights, runtimes, credentials, and new experiment results are excluded from Git.
The small included baseline dataset contains selected non-private answers and
measurements. New runs go in `results`; generated code is recorded, not executed.

Primary references are in [SOURCES.md](SOURCES.md) and at the end of infrastructure
chapters. [ERRATA.md](ERRATA.md) records corrections to the original draft.
