# Understanding local LLM inference and hosting

This guide uses one running example: a local Qwen3.5-4B model served by Ollama
on an RTX 4050 Laptop GPU with about 6 GiB VRAM. The principles generalize;
the measured numbers do not automatically transfer to another machine.

Read each section as: **mental model -> mechanics -> implications -> experiment**.
Primary references are indexed in [SOURCES.md](SOURCES.md).

## 1. What are you actually hosting?

You are not just opening a model file. You are running a service:

```text
client -> HTTP server -> scheduler -> tokenizer/chat template
       -> model runner -> CPU/GPU kernels -> sampler -> streamed response
```

**Model weights** are the learned numerical parameters. **The tokenizer**
maps text to token IDs. **The chat template** encodes roles and boundaries
so the model can distinguish system instructions, user text, and its answer.
**The runner** allocates memory and executes the model. **The scheduler**
decides which requests can run now and which must wait.

An analogy: a restaurant has a recipe book (weights), order tickets (tokens),
a chef (the runner), a kitchen (the GPU), a host (the scheduler), and serving
staff (the API). A faster recipe does not fix a queue at the door.

Ollama provides packaging, model management, an API, and execution backends.
llama.cpp provides inference machinery and tools with lower-level controls.
LM Studio offers a desktop-oriented workflow. vLLM and SGLang emphasize serving
features such as scheduling and batching, but hardware, OS, model, and version
support must be checked. Switching servers is not automatically a speed upgrade.

GGUF is a file format, not a model architecture or a precision level.
CUDA is a GPU programming platform, not a guarantee that every model operation
is efficient. A model can be fully on GPU and still run slowly.

**On this laptop:** the teaching deployment uses port 11435 to avoid assuming
ownership of an existing Ollama instance on port 11434. Model storage belongs
to the server process, not the CLI client. Pointing a CLI at another port does
not magically give that server access to the same model store.

## 2. Tokens, prefill, and decode

A token can be a word, part of a word, punctuation, whitespace, or another unit.
There is no universal words-to-tokens ratio, especially for code and multilingual
text. Use the tokenizer or runtime-reported token counts.

For a causal language model, the central loop is:

```text
encode prompt -> process prompt -> predict next token
              -> append token -> predict another -> ... -> stop
```

**Prefill** processes the input prompt and initializes the model's state.
Many prompt positions can be computed together using matrix operations.
The causal mask prevents a position from reading future tokens.
Backends can split a long prompt into chunks; it need not all be processed
in one monolithic allocation.

**Decode** produces subsequent tokens, usually one at a time per sequence.
The next token depends on the previous result. A model cannot simply generate
the entire answer in parallel without a different algorithm.

Imagine reading a 20-page question once, then writing the answer one word at a
time. Reading faster and writing faster are different optimizations.

The context budget includes the system message, chat-template tokens,
conversation history, new prompt, generated reasoning, and answer. For example,
3,000 input tokens plus 1,500 output tokens do not fit in a 4,096-token budget.
Exact overflow/truncation behavior is backend-dependent; inspect it rather
than silently losing early instructions.

**Thinking mode:** reasoning tokens consume time and memory too. A server can
emit a thinking field before any visible answer. Distinguish time to the first
stream event, first generated content, and first answer content. Our baseline
requested thinking off.

## 3. Attention and the KV cache, from a small example

For a simplified single attention head:

```text
q = current hidden state projected through Wq
K = previous/current hidden states projected through Wk
V = previous/current hidden states projected through Wv
attention(q, K, V) = softmax(q K^T / sqrt(head_dim)) V
```

Q asks a question, K contains labels used to judge relevance, and V contains
the information to blend. These are learned vectors, not literal text labels
or a conventional database key-value store.

Suppose two tokens receive attention scores 2 and 0, and their scalar values
are 10 and 20:

```text
softmax([2, 0]) = approximately [0.881, 0.119]
output = 0.881 * 10 + 0.119 * 20 = approximately 11.19
```

The output is a weighted blend, not necessarily a selection of one token.
Actual models have multiple heads, layers, residual connections, normalization,
position information, and feed-forward networks.

### Why caching is valid

In ordinary causal inference, an earlier token's representation does not gain
information from future tokens. With the same prefix and positions, its K/V
can be retained instead of recomputed each time. New queries consult those
stored vectors. Ordinary decoding does not need to retain all old queries.

At 100 tokens, the cache avoids rewriting 100 index cards. It does not mean
the next query never reads those cards.

For N steps, recomputing K and V for every prefix processes:

```text
2 * (1 + 2 + ... + N) = N * (N + 1) token-row projections
```

Caching processes `2 * N` token rows. But full attention still compares each
new query to its available history. Its cumulative decode attention work
remains quadratic in sequence length. The linear savings above apply to K/V
projection work, not the entire model.

Run `01_why_kv_cache.py`. It feeds predetermined embeddings into a toy head
and compares cached/uncached results. Its CPU timing is not a Qwen benchmark.

### Not distillation

Distillation trains a student using a teacher's outputs or representations.
The student often is smaller, but a different architecture is not mandatory.
It can outperform the teacher on a narrow objective; a fixed quality loss is
not part of the definition.

Caching changes execution, not training. Cache quantization introduces numerical
approximation. Weight quantization changes stored parameter precision. GQA is
a learned architecture choice. None of these terms is interchangeable.

## 4. Where the memory goes

Use a budget, not just the model's download size:

```text
GPU memory approximately =
    resident weights
  + conventional attention KV cache
  + recurrent state, if the architecture has it
  + temporary activations and compute buffers
  + runtime/allocator overhead
  + other GPU users
```

System RAM is a separate budget: Windows, applications, host-side model data,
memory-mapped pages, transfer buffers, and CPU-offloaded work compete there.
VRAM and RAM are not one interchangeable fast pool. A page file is not a
substitute for GPU memory, and low "free RAM" does not alone prove active paging.

### Units matter

`1 GB = 1,000,000,000 bytes`. `1 GiB = 1,073,741,824 bytes`.
`1 MiB = 1,048,576 bytes`. The observed 6,141 MiB GPU capacity is about
5.997 GiB, not 6.141 GiB.

The Ollama package was about 3.39 decimal GB, or 3.16 GiB. Package size,
runner-reported residency, and driver-reported usage measure different things.
Some packages contain components not simultaneously resident during text-only
inference. Do not infer exact weight usage by subtracting unrelated counters.

### Conventional attention cache formula

For equal K and V head dimensions and unshared independent sequences:

```text
bytes = 2 * layers * KV_heads * head_dim * tokens * bytes_per_value * sequences
```

For a model with different layer types or dimensions, calculate each relevant
layer's storage and add the results. Allocation granularity, padding, sliding
windows, shared prefixes, paged caches, and compressed latent representations
can change the actual amount. The formula is not universal for every architecture.

For Qwen3-8B at 8,192 tokens and one sequence:

```text
FP16: 2 * 36 * 8 * 128 * 8192 * 2 = 1,152 MiB = 1.125 GiB
Q8_0 storage: 1,152 * (34/32) / 2 = 612 MiB
```

The extra 2 bytes in a 34-byte block store an FP16 scale for 32 quantized values.
Ignoring scales understates memory.

### GQA: sharing keys and values

MHA uses separate K/V heads for each query head. MQA shares one K/V head across
all query heads. GQA shares K/V within groups of query heads.

With all else equal, 32 KV heads versus 8 means four times the cache.
But two models with different layer counts cannot be compared solely by their
KV-head counts.

Two researchers can share the same books and still look up completely different
facts. Shared keys do not force identical attention. Run `04_gqa_explained.py`
to see exactly that. Random-weight correlations cannot establish model quality.

### Your Qwen3.5-4B is hybrid

Its text configuration has 32 layers: 24 Gated DeltaNet layers and 8 full-attention
layers. The full-attention layers have 4 KV heads with head dimension 256.

The recurrent layers maintain evolving state rather than the same growing
per-token K/V arrays used by full attention. Think of a compact, continually
updated summary versus a shelf with one card per token. This analogy does not
mean the recurrent state is a human-readable summary or cost-free.

For the full-attention portion only:

| Context, one sequence | FP16 cache | Q8_0 storage arithmetic |
| --- | ---: | ---: |
| 4,096 | 128 MiB | 68 MiB |
| 8,192 | 256 MiB | 136 MiB |
| 16,384 | 512 MiB | 272 MiB |
| 32,768 | 1,024 MiB | 544 MiB |

This excludes recurrent state, checkpointing strategy, and other allocations.
It also does not establish runtime support for quantized caches on this hybrid.

At 4K, switching those K/V arrays to Q8 could save about **60 MiB**, not gigabytes.
That is why "always enable Q8" is not a good first optimization for our deployment.
Run `02_kv_cache_size.py --context 8192 --sequences 2` and inspect the change.

## 5. Quantization: what changes and what does not

FP16 and BF16 both use two bytes per value but have different precision/range
tradeoffs. INT8 stores integer codes; a scale maps them back to approximate
real values. FP4 and four-bit integer schemes are not the same representation.

For a simple block, if scale is 0.1, a value 0.37 might become integer 4 and
reconstruct as 0.4. You save storage but introduce error. Per-block scales,
zero points, outlier handling, and mixed tensor precision change the result.

`Q4_K_M` describes a particular weight-quantization recipe, not "every value
in the entire model uses exactly four bits." Metadata, unquantized tensors,
and mixed-precision choices affect file size. `q8_0` for the cache is a separate
setting; you can run Q4 weights with an FP16 cache.

Lower precision may reduce memory traffic, but decoding must unpack/dequantize
the data or use suitable kernels. Faster is not guaranteed. A poorly supported
format can lose to a larger, better-supported one.

Small vector error can alter logits enough to change a chosen token. That token
changes the next prefix, so later answers can diverge. Conversely, different
text can still be equally correct. Cosine similarity of one toy attention
output is not a language-model accuracy score.

Run `03_kv_quant_error.py`. It explicitly labels its low-bit example as a
toy format, includes multiple queries, and makes no end-to-end quality claim.
The all-zero block case illustrates why quantization needs careful arithmetic.

For a real comparison, evaluate the same tasks and include exact requirements,
long-context retrieval, factual correctness, code edge cases, and formatting.
Do not use "sounds fluent" as your only scoring rule.

## 6. FlashAttention: avoid an enormous temporary worksheet

For unchunked full prefill, materializing attention scores produces an
`N x N` matrix per head. At N=8,192, FP32 storage for one such matrix is
256 MiB. This is temporary attention workspace, not the KV cache.

FlashAttention tiles the computation and uses an online softmax so it need not
write the full matrix to GPU memory. Real implementations use fast on-chip
memory, fused kernels, and careful scheduling. A NumPy loop illustrates the
math but not those hardware optimizations.

For scores arriving in blocks, keep:

```text
m = largest score seen
l = sum of exp(score - m)
o = sum of exp(score - m) * value
```

When a new block raises the maximum to `m_new`, rescale the old accumulators:

```text
correction = exp(m - m_new)
l_new = correction * l + sum(exp(new_scores - m_new))
o_new = correction * o + sum(exp(new_scores - m_new) * new_values)
answer = o_new / l_new
```

This is like recomputing the units on your running tally, not discarding the
previous blocks. The mathematical result is unchanged; floating-point order
can cause small numerical differences.

Run `05_flash_attention.py`. Its storage tables count score arrays/tiles only.
Actual kernels also need Q/K/V tiles, accumulators, and occupancy-dependent
workspace. Total model memory still grows with the context and other states.
Inference layers need not retain every score matrix simultaneously.

FlashAttention and quantized KV support are runtime- and architecture-specific.
Current Ollama documents quantized KV with FlashAttention. Current llama.cpp can
auto-enable required FlashAttention or report unsupported quantized V settings.
Do not repeat the claim that all such flags are silently ignored. Check the
installed version's logs and effective configuration, not just the environment.

## 7. Read the actual baseline correctly

Our first run reported:

| Request | First output | Model load | Generation | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic, initial load | 101.96 s | 60.55 s | 1.16 tok/s | 105.41 s |
| Same arithmetic, warm | 0.19 s | 0.004 s | 17.59 tok/s | 0.42 s |
| Coding, warm | 0.40 s | 0.002 s | 10.25 tok/s | 8.78 s |
| Explanation, warm | 0.56 s | 0.003 s | 14.15 tok/s | 11.37 s |

Ollama durations are in nanoseconds. Convert by dividing by `1e9`.

```text
decode tokens/s = eval_count / (eval_duration / 1e9)
prefill tokens/s = prompt_eval_count / (prompt_eval_duration / 1e9)
```

The first request reported 28 prompt tokens at about 0.677 tokens/s:
approximately **41.37 seconds of reported prefill**, in addition to model
loading. We cannot attribute all 102 seconds to downloading or loading.
Disk caching, lazy initialization, kernel setup, and memory pressure are
possible contributors, not established diagnoses.

The four-token arithmetic answer is too short to represent steady decode speed.
Combining coding and explanation as total generated tokens divided by total
decode time gives about 12.4 tokens/s. This still is only two short samples.

Ollama reported 100% GPU residency. NVIDIA snapshots reported 3,829 MiB used
out of 6,141 MiB after each request. Neither tells us the memory peak.
"100% GPU" in `ollama ps` describes placement, not utilization, bandwidth,
kernel efficiency, or freedom from CPU overhead.

Run `06_read_inference_results.py` to reconstruct the durations offline.

### Quality also needs criticism

The arithmetic answer was correct. The generated palindrome function was
plausible, but its assertions included no negative example. It was not executed.
The explanation used exactly 120 whitespace-separated words despite a request
for **under** 120. It also overstated architecture change as a requirement
of distillation. A fluent explanation can contain subtle inaccuracies.

## 8. Design a trustworthy experiment

Write a question before changing anything. Examples:

| Question | Hold fixed | Change | Record |
| --- | --- | --- | --- |
| Does keeping the model loaded help? | Model, prompt, context | Loaded vs unloaded | Load time, first output |
| Does longer context cost memory? | Model, precision, concurrency | Configured context, then separately actual prompt length | Memory, prefill, decode |
| Does Q8 help this model? | Artifact, prompts, context | Effective cache precision | Memory, latency, quality |
| Does concurrency help a service? | Model, task distribution | Simultaneous requests | Aggregate throughput, per-user latency, errors |
| Is a larger quant worth it? | Evaluation tasks | Exact weight artifact | Quality, placement, first output, decode |

Keep a stable prompt suite and record model digest, runtime, driver, settings,
power conditions, and relevant background workloads. Mutable tags can change.
Temperature zero and a seed reduce variation but do not guarantee identical
results across kernels or software versions.

Separate a warm-up from measured repetitions. Use at least several repetitions
for small comparisons and alternate A/B order to reduce drift. Report the median
and spread. Do not estimate a meaningful p95 from four requests.

Prefix reuse is a confounder: repeating the same prompt may skip prefill work.
Report reused-prefix and fresh-prefix cases separately. Unloading the model
does not flush Windows' filesystem cache or reproduce a reboot-cold machine.

Measure actual input length, not just `num_ctx`. Reserving 32K context and
sending a ten-token question does not measure 32K prompt processing.

Define an acceptable answer before benchmarking. A fast truncated answer or
an incorrect JSON object is not a performance win. Pair speed with a task score.

## 9. Why a fully loaded GPU may still be slow

Decode often performs matrix-vector-like operations for one sequence. It may
spend much of its time fetching weights rather than doing arithmetic. Prefill
has more matrix-matrix work and can reuse weights across token positions.

A simplified weight-bandwidth bound is:

```text
max tokens/s approximately <= available bytes/s / weight bytes read per token
```

Assume, purely for illustration, 3 GiB read per token and 100 decimal GB/s:
the weight-only bound is about 31 tokens/s. It omits KV reads, compute, recurrent
updates, unpacking, kernel launch overhead, CPU work, and transfers. It is not
a prediction for your GPU. Run `07_performance_budget.py`.

A useful mental model is that execution is constrained by the larger of compute
time and memory-transfer time, with further costs and imperfect overlap.
Do not sum independently quoted peak FLOPS and bandwidth into a speed promise.

For MoE models, only some experts activate per token, but all required experts
must be stored or fetched. Total parameters determine storage pressure; active
parameters help describe work. Routing, expert reuse, and offloading matter.

CPU offloading can add host computation and data transfer. Its cost depends on
the amount and placement of work; it is not a universal five- or tenfold cliff.
More system RAM can help avoid paging but does not create GPU bandwidth.

### Optimize the largest measured cost first

1. Correct the backend or unexpected CPU placement before exotic tuning.
2. If startup dominates, preload/keep the model resident when appropriate.
3. If answers are long, use a reasonable output budget and task-specific prompts.
4. If prefill dominates, reduce irrelevant history and evaluate prefix reuse.
5. If memory is limiting, reduce context or concurrency, then evaluate quantization.
6. Only then compare backend/kernel settings or different model artifacts.

At 12 tokens/s, 240 generated tokens take about 20 seconds even with an immediate
start. A useful 120-token answer can nearly halve that time. Streaming improves
perceived responsiveness; it does not eliminate the token computation.

Power mode, clocks, temperature, battery operation, and background GPU workloads
can matter. Observe them. Do not assume a specific power limit or a fixed
"throttle after ten minutes" rule.

## 10. From one chat to a hosted service

Single-user latency and total server throughput are different objectives.

Suppose one request produces 12 tokens/s. Two concurrent requests might each
produce 8 tokens/s: total throughput rises to 16, but each user waits longer.
Or both metrics might get worse if memory pressure causes offloading.
Those figures illustrate the tradeoff, not measured behavior here.

Batching can reuse weight reads across multiple sequences. Continuous batching
lets a server admit and retire requests as generation proceeds. Prefill-heavy
requests can interfere with decode; scheduling and chunked prefill affect fairness.
Features differ across backends and versions.

With no prefix sharing, conventional cache storage grows approximately with
the sum of active sequence lengths. Configured parallel slots can reserve
additional memory even before all slots are busy. Four 8K sequences are not
the same memory budget as one 8K sequence.

A queue absorbs bursts, not unlimited demand. If requests arrive faster than
the service can finish them, waiting time grows. Bigger queues can hide
overload while making latency unacceptable. Set request limits, timeouts, and
backpressure; handle overload responses explicitly.

For the teaching deployment, one parallel request is a deliberate baseline.
Do not immediately set concurrency to CPU core count. CPU threads and GPU
request slots are different controls.

## 11. Operate the server deliberately

The original deployment lives outside this repository:

```text
C:\Projects\Qwen-Local
  runtime          portable runner
  models           weight files
  .home            local application/key data
  results          recorded private runs
  Start-Server.ps1
```

The server owns environment settings. Setting `OLLAMA_MODELS` or a cache type
in a client terminal does not alter an already-running server.

`$env:NAME = 'value'` affects the current PowerShell process and its children.
`setx` changes future processes, not the current server, and persists outside
the experiment. Prefer process-local settings for controlled learning.
Restart the intended server when changing server-level settings.

Our launcher explicitly sets FP16 cache and one parallel request. An external
environment assignment will be overwritten by those lines. Inspect the launcher
before claiming a flag took effect.

Keep-alive controls how long the loaded model remains resident, not whether
the HTTP server process survives. A model can unload while the server stays up.
A server launched in an attached assistant session can terminate when that
session ends; run the provided server command in your own terminal for continued use.

Use logs to confirm runner selection, placement, context, and supported features.
For monitoring, distinguish allocation from active GPU utilization and use
timestamped samples over the workload. Power and temperature samples explain
conditions, not necessarily causation. Never disable security software simply
to make a benchmark number larger.

Before installing a runtime, locate the existing executable, check its version,
query its endpoint, and locate its model store. Our initial setup unnecessarily
downloaded another runtime because discovery was incomplete. Model downloads
should be reusable; avoid paying that cost for every experiment.

## 12. Hosting safely and choosing the next skill

Keep the raw inference API on loopback for a personal laptop. Local APIs often
assume a trusted local client; CORS is not authentication. Do not expose a bare
server on all interfaces or through a public tunnel for convenience.

For a real multi-user deployment, put authentication, TLS, request/body limits,
timeouts, rate limits, and appropriate access logging in front of it. Protect
logs: prompts, retrieved documents, and generated answers may contain private
information. Local execution does not make an application automatically private
if other components send data to cloud services.

Generated text and code are untrusted output. Tool-calling capability does not
authorize execution. Use allowlists, narrow permissions, explicit approvals,
and isolation for tools. Do not run arbitrary model-produced shell commands.

Useful next topics, after a reliable baseline:

| Technique | What it addresses | Important limitation |
| --- | --- | --- |
| Prefix caching | Repeated identical input prefixes | Requires compatible state/positions; consumes memory |
| Paged KV allocation | Fragmentation and request scheduling | Does not erase live token storage |
| RAG | Supplying relevant external facts | Retrieval adds work; irrelevant chunks inflate context |
| Speculative decoding | Verify several proposed tokens together | Draft cost and acceptance rate decide the win |
| Structured output | Constrain response syntax | Valid JSON can still contain false information |
| Multi-GPU serving | Capacity or parallel computation | Communication overhead can outweigh gains |
| Fine-tuning/LoRA | Adapt behavior to a task | Training is not a remedy for serving bottlenecks |

There is no single "best optimization." A good result is a configuration that
meets a stated quality target and latency target within your memory, power,
and operational constraints. That is what the labs teach you to measure.
