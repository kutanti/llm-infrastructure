# Visual index

Each visual sits beside the equations or system behavior it explains.
The diagrams use GitHub's native Mermaid rendering. Animation panels are
collapsed by default; the static PNGs and captions can be read without motion.
Open an animation panel to play its looping GIF; close it to hide the motion.

## Step-by-step animations

| Mechanism | Explanation | Without motion | Animation file |
| --- | --- | --- | --- |
| Gradient descent | [Scalar weight update](../foundations/02-learning-from-mistakes.md#begin-with-one-weight) | [PNG](animations/gradient-descent.png) | [GIF](animations/gradient-descent.gif) |
| KV reuse | [Prefill versus decode](../course/05-kv-cache-and-quantization.md#1-follow-one-conversation) | [PNG](animations/kv-cache.png) | [GIF](animations/kv-cache.gif) |
| Quantization | [Rounding and reconstruction](../course/04-weight-and-activation-quantization.md#1-affine-quantization) | [PNG](animations/quantization-grid.png) | [GIF](animations/quantization-grid.gif) |
| Continuous batching | [Decode slot reuse](../course/06-serving-and-distributed-systems.md#62-fixed-batching-continuous-batching-and-chunked-prefill) | [PNG](animations/continuous-batching.png) | [GIF](animations/continuous-batching.gif) |

The scalar loss and quantization positions are calculated from the equations.
The cache and scheduler panels are state-transition schematics, not recordings
of a GPU. Their frame durations are chosen for reading, not proportional to
execution time.

## Structural diagrams

- [Text, token IDs, embeddings, and logits](../foundations/01-text-and-representations.md)
- [Pre-norm transformer residual paths](../foundations/03-inside-a-transformer.md#one-decoder-block)
- [GPU memory and host/device transfers](../course/01-gpu-systems.md#12-memory-capacity-and-bandwidth-are-different-constraints)
- [LoRA's frozen and trainable branches](../course/02-training-and-fine-tuning.md#5-lora-train-an-update-instead-of-the-whole-matrix)
- [QLoRA's stored base and computation](../course/02-training-and-fine-tuning.md#6-qlora-frozen-low-bit-base-plus-trainable-adapters)
- [Teacher/student losses and gradient flow](../course/03-distillation.md#2-temperature-and-soft-targets)
- [Logical cache blocks and shared physical pages](../course/05-kv-cache-and-quantization.md#3-logical-length-versus-allocated-capacity)
- [FP16, Q8_0, and Q4_0 cache storage](../course/05-kv-cache-and-quantization.md#5-quantizing-k-and-v)
- [Inference replication](../course/06-serving-and-distributed-systems.md#data-parallelism-and-ddp), [tensor partitioning](../course/06-serving-and-distributed-systems.md#tensor-parallelism), and [pipeline stages](../course/06-serving-and-distributed-systems.md#pipeline-parallelism)

## Rebuild the images

The checked-in files can be viewed without Python or extra dependencies.
To change them, edit [the renderer](../tools/render_visuals.py), then run these
commands from the repository root:

```powershell
python -m venv tools\.venv
.\tools\.venv\Scripts\python.exe -m pip install -r tools\requirements-visuals.txt
.\tools\.venv\Scripts\python.exe tools\render_visuals.py
```

Pillow is an optional build dependency, isolated from the NumPy lessons.
The renderer uses its scalable default font and a shared palette per animation;
there are no external fonts, stock images, model calls, or rendering services.
Each GIF is 1000 x 570 pixels with a corresponding static PNG. Pillow versions
may change rasterization or compression, so regenerated files need not be
byte-identical across versions. Mermaid source is edited directly in Markdown.
