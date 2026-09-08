# LLM infrastructure course

This course follows a model through its working life:

```text
learn the model computation
    -> understand the GPU that executes it
    -> adapt or distill the weights
    -> choose numerical representations
    -> manage request state
    -> serve and measure the resulting system
```

The goal is to reason about a training or serving system, not memorize flags for
one runtime. The original 6 GB laptop deployment is one case study.

## Prerequisites

You should be comfortable reading Python arrays and functions. Start with the
[foundations chapters](../foundations/README.md) for vectors, tokenization,
embeddings, probabilities, gradients, and transformer data flow.

You do not need a GPU to work through the mathematical examples. NumPy is the
only required package for the included training demonstrations.

## Sequence

| Unit | Read | What you should be able to do |
| --- | --- | --- |
| 0. Model foundations | [Four introductory chapters](../foundations/README.md) | Trace text through logits, loss, gradients, and generation |
| 1. GPU systems | [GPU execution and memory](01-gpu-systems.md) | Separate capacity, bandwidth, compute, launches, and transfers |
| 2. Training and fine-tuning | [SFT, LoRA, and QLoRA](02-training-and-fine-tuning.md) | Construct targets, budget a step, and evaluate an adapter |
| 3. Distillation | [Teacher signals and student objectives](03-distillation.md) | Implement soft-target learning and measure transfer independently of correctness |
| 4. Quantization | [Weights, activations, and kernels](04-weight-and-activation-quantization.md) | Explain scales, groups, calibration, precision, and artifact compatibility |
| 5. Request state | [KV cache and cache quantization](05-kv-cache-and-quantization.md) | Calculate storage, reason about prefix reuse, and design a cache experiment |
| 6. Serving at scale | [Scheduling and distributed systems](06-serving-and-distributed-systems.md) | Connect batching, queueing, parallelism, and latency objectives |

Do the [practical workbook](WORKBOOK.md) as you go. It contains calculations,
small training experiments, and larger capstones with explicit deliverables.

## Included experiments

From the repository root:

```powershell
python -m course.demos.lora
python -m course.demos.distillation
python -m course.demos.quantization
```

The LoRA experiment learns a low-rank update while preserving a frozen base.
The distillation experiment implements the temperature-scaled soft-target
gradient. The quantization experiment measures code/scale storage and the
effect of grouping on output error.

These are small numerical experiments, not downloads or fine-tuning runs of
Qwen. They expose operations that a real training framework hides behind APIs.

The earlier examples remain useful:

```powershell
python 02_kv_cache_size.py --context 32768 --sequences 2
python 03_kv_quant_error.py
python 05_flash_attention.py
python 06_read_inference_results.py
```

For live local inference, [SETUP.md](../SETUP.md) uses an existing Ollama installation.
The [laptop case study](../GUIDE.md) and [inference labs](../LABS.md) provide measured
examples and a client that records request timing.

## What requires additional hardware or software

The included CPU experiments run as supplied. A real GPU SFT/QLoRA job requires
a compatible checkpoint, dataset, CUDA-capable training framework, and enough
memory. Chapter 2 provides the full experiment protocol and official framework
references; this repository does not yet supply a version-pinned GPU training
runner. Distributed training/serving capstones require suitable multi-GPU resources.

Keep those boundaries visible in your own work. A numerical explanation, a
working CPU example, a GPU training run, and a production service are different
deliverables.

## Completion criteria

Given an unfamiliar checkpoint and workload, you should be able to:

1. Identify architecture, tokenizer, weight representation, and runtime support.
2. Estimate weights, optimizer state, activations, and cache separately.
3. Decide whether the goal needs prompting, retrieval, fine-tuning, distillation,
   numerical compression, or serving changes.
4. Construct a comparison with fixed data, effective configuration, and independent
   quality and performance measurements.
5. Explain where each millisecond and major allocation comes from, or identify
   the missing measurement rather than inventing a cause.

Use the capstone artifacts to demonstrate those abilities.
