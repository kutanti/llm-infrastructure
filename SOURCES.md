# References

Primary sources consulted on 2026-09-07. Web documentation
and `main`/`master` branches are mutable; check the installed runtime's behavior.
Publisher benchmark scores are not our measurements of quantized inference.

## Model architecture

- [Qwen3.5-4B official configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json):
  32 text layers, 8 full-attention layers, 4 KV heads, head dimension 256,
  and 24 linear/recurrent layers. This is the source of the hybrid cache example.
- [Qwen3.5-4B official model card](https://huggingface.co/Qwen/Qwen3.5-4B):
  model overview, deployment information, and publisher evaluations.
- [Qwen3-8B configuration](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
  and [Qwen3-4B configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json):
  conventional cache examples with 36 layers, 8 KV heads, dimension 128.
- [Original GLM-4-9B-Chat configuration](https://huggingface.co/zai-org/glm-4-9b-chat/blob/main/config.json):
  40 layers, 2 multi-query groups, KV dimension 128. This refers to the original
  checkpoint, not later releases with similar names.
- [GLM-4.7-Flash official model card](https://huggingface.co/zai-org/GLM-4.7-Flash):
  30B-A3B MoE, not an 8B dense model.
- [Llama 2 paper](https://arxiv.org/abs/2307.09288):
  architecture context for the original 7B conventional attention example.

## Attention and storage mechanics

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762):
  scaled dot-product and multi-head attention.
- [GQA paper](https://arxiv.org/abs/2305.13245):
  grouped-query attention and trained-model comparisons.
- [FlashAttention paper](https://arxiv.org/abs/2205.14135):
  IO-aware exact attention with tiling. "Exact" describes the mathematical
  attention operation, not bit-for-bit identity across floating-point kernels.
- [llama.cpp block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
  and [reference quantization routines](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c):
  Q8_0 blocks store 32 signed bytes plus a 2-byte scale; Q4_0 blocks store
  16 packed bytes plus a 2-byte scale. The toy4 demo is not this encoding.
- [llama.cpp context implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp):
  quantized V cache / FlashAttention compatibility handling.

## Serving and API

- [Ollama FAQ](https://docs.ollama.com/faq):
  environment settings, GPU placement, keep-alive, concurrency, FlashAttention,
  cache types, model storage, and local-only mode.
- [Ollama chat API](https://docs.ollama.com/api/chat):
  streaming responses, thinking controls, token counts, and durations.
- [Ollama running-model API](https://docs.ollama.com/api/ps):
  loaded-model metadata.
- [Ollama v0.33.3 release](https://github.com/ollama/ollama/releases/tag/v0.33.3):
  runtime version used for the original run.
- [Qwen3.5-4B Ollama manifest](https://registry.ollama.ai/v2/library/qwen3.5/manifests/4b):
  package metadata. The tag is mutable; the baseline records its model digest.
- [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180):
  memory management and serving throughput concepts.
- [Speculative decoding paper](https://arxiv.org/abs/2211.17192):
  draft-and-verify acceleration; benefits depend on acceptance and costs.

## What was measured here

`data\qwen-baseline.json` contains selected observations and answers from the
original deployment's `baseline-20260907-222303.json`. The original, larger report
remains outside this repository in the local deployment's results directory.
It included model metadata and GPU snapshots.

The following were not established: sustained throughput, peak memory, GPU
bandwidth, power limit, a cause for the slow first request, the effective
FlashAttention state, comparative model quality, or an optimized setting.

The offline scripts illustrate individual operations; their timings are not
measurements of the deployed model.
