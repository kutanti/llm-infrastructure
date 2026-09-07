# Why did the first answer take 102 seconds?

We loaded Qwen3.5-4B on a laptop with an RTX 4050 and asked:

> What is 17 * 23? Reply with only the integer.

The answer was `391`. Getting the first character took 101.96 seconds.
Repeating the question took 0.19 seconds.

The model and hardware did not change between those requests. Before replacing
the model or adjusting GPU settings, we need to account for that difference.

## 1. Split the waiting time

An inference request passes through several stages:

```text
wait for a slot
    -> load the model if needed
    -> process the prompt (prefill)
    -> generate tokens (decode)
    -> deliver the response
```

The server also tokenizes the input and applies a chat template. The template
marks user messages, assistant messages, and other boundaries in the format the
model expects. Tokens are the model's input units; a token can be a word,
part of a word, or punctuation. Count them with the tokenizer or runtime,
not by assuming a fixed number of characters per token.

Ollama reports separate load, prompt-evaluation, and generation durations.
Our client records elapsed time until the first output and the end of the stream.

| Request | First output | Load | Prefill | Decode | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arithmetic, first load | 101.96 s | 60.55 s | 41.37 s | 3.45 s | 105.41 s |
| Arithmetic, warm | 0.19 s | 0.004 s | 0.18 s | 0.23 s | 0.42 s |
| Python function, warm | 0.40 s | 0.002 s | 0.28 s | 8.39 s | 8.78 s |
| Explanation, warm | 0.56 s | 0.003 s | 0.47 s | 10.81 s | 11.37 s |

The prompt and decode durations above are reconstructed from the recorded
token counts and rates. Run `06_read_inference_results.py` to see the calculation.

The first request spent about a minute loading, but that is only part of the
delay. It also reported 41 seconds processing a 28-token prompt. That is an
investigation target: the record does not tell us how much came from lazy
initialization, kernel setup, memory pressure, or other work inside that stage.

The later requests spent almost no time loading. Their wall time was mostly
generation. Keeping the model resident addresses the first problem; it will
not make an already-loaded model write a long answer faster.

### Three meanings of "first output"

A streaming endpoint can send metadata, then reasoning text, then an answer.
Record these separately:

| Metric | What the client has received |
| --- | --- |
| First event | Any parsed stream event |
| First generated output | Reasoning or answer text |
| First answer | Text in the answer field |

Our baseline requested thinking off. With thinking on, an early stream event
can make a request look responsive while the user is still waiting for an answer.
The live client records all three timestamps.

## 2. What the GPU does during prefill and decode

During **prefill**, the model processes the prompt. Many prompt positions can be
handled together using matrix operations. A causal mask prevents each position
from reading tokens that come after it.

During **decode**, the model predicts the next token, appends it to the sequence,
and repeats. For a single request, the next step depends on the previous result.
This gives the GPU much less parallel work than processing many prompt positions
at once.

For example, summarizing a long document into one sentence may spend most of its
time in prefill. Writing a thousand-word explanation from a short question may
spend most of its time in decode. Both can be described as "slow," but shortening
the prompt helps only one of them.

A rough estimate for answer time is:

```text
time to finish = time to first answer + remaining answer tokens / decode rate
```

At 12 tokens/s, 239 tokens after the first take about 20 seconds. Streaming lets
you read while they arrive; it does not remove that computation.

Ollama returns durations in nanoseconds:

```text
decode tokens/s = eval_count / (eval_duration / 1e9)
prefill tokens/s = prompt_eval_count / (prompt_eval_duration / 1e9)
```

Use the rate for the stage you are investigating. Averaging a cold four-token
request with a warm 500-token request produces a number with little practical use.

## 3. Why the model remembers keys and values

Consider a single attention head:

```text
q = current hidden state multiplied by Wq
K = available hidden states multiplied by Wk
V = available hidden states multiplied by Wv
output = softmax(q K^T / sqrt(head_dim)) V
```

The query `q` is compared with the keys `K` to get relevance scores. Softmax
turns those scores into weights. The output is a weighted blend of the values `V`.
These are learned vectors, rather than labels or text stored in a database.

With two scores, 2 and 0, and scalar values 10 and 20:

```text
softmax([2, 0]) = approximately [0.881, 0.119]
output = 0.881 * 10 + 0.119 * 20 = approximately 11.19
```

Now append another token. In causal inference, the earlier positions did not
see that future token. Their keys and values are still usable, so recomputing
them would repeat work. The **KV cache** retains them. Each new query reads
the available cache and adds the new token's keys and values.

For 400 steps, rebuilding both K and V for every prefix processes:

```text
2 * (1 + 2 + ... + 400) = 160,400 token-row projections
```

Keeping the previous projections reduces this to `2 * 400 = 800`.
The first Python lesson demonstrates that difference with predetermined random
embeddings. It compares the attention outputs, not generated language.

The cache removes repeated projections, not the need to consult history.
A new full-attention query still compares against the retained keys. Across
N decode steps, those comparisons grow as `1 + 2 + ... + N`.

### Caching is not distillation

Caching reuses work during inference. Distillation trains a student from a
teacher's outputs or representations. The student often is smaller, but it need
not use a different architecture.

Two other terms will matter below: **weight quantization** stores model parameters
at lower precision; **cache quantization** stores cached state at lower precision.
Neither is another name for training a student.

## 4. Account for the 6 GB of VRAM

The driver reported 6,141 MiB of total GPU memory. After each baseline request,
it reported 3,829 MiB used. Ollama reported the model as `100% GPU`.

Those counters have different meanings. Ollama describes model placement;
the driver includes other allocations. Neither is a measurement of the peak
memory used during a request.

The budget looks like this:

```text
resident model weights
+ full-attention KV cache
+ recurrent state, for architectures that use it
+ temporary activations and compute buffers
+ runtime and allocator overhead
+ other GPU applications
```

The model download was about 3.39 GB. That does not give us the rest of this
budget. A package can include components not used simultaneously, and runtime
buffers depend on the workload.

Use consistent units: a GB is one billion bytes; a GiB is `1024^3` bytes.
6,141 MiB is about 5.997 GiB. System RAM is another budget entirely. CPU-offloaded
work, host buffers, Windows, and other applications compete there; a large page
file does not create more VRAM.

### Calculate a conventional cache

For layers with equal K/V head dimensions:

```text
bytes = 2 * layers * KV_heads * head_dim * tokens * bytes_per_value * sequences
```

This counts the K/V tensors for independent sequences. It excludes allocator
padding and temporary workspace. Architectures using sliding windows, compressed
latent state, or recurrent layers require different accounting.

Qwen3-8B provides a conventional example:

```text
2 * 36 layers * 8 KV heads * 128 dimensions * 8192 tokens * 2 bytes
= 1,152 MiB in FP16
```

Q8_0 stores 32 values in 34 bytes: 32 integer codes and a two-byte scale.
For the same tensor shapes:

```text
1,152 MiB * (34/32) / 2 = 612 MiB
```

This is why an estimate using exactly one byte per Q8 value is slightly low.

### Share K/V heads, not queries

In multi-head attention (MHA), each query head has its own K/V head.
Grouped-query attention (GQA) lets several query heads share one K/V head.
Multi-query attention (MQA) uses a single K/V head for all query heads.

Hold layer count, dimensions, context, and precision fixed: reducing 32 KV heads
to 8 cuts their storage by four. The query heads still differ. Two queries can
use the same keys and assign them different weights, as lesson 4 demonstrates.

GQA is part of the learned architecture. Changing a flag does not convert an
arbitrary MHA checkpoint into a trained GQA checkpoint.

## 5. Our model has a smaller attention cache than that

Qwen3.5-4B has 32 text layers, but only **8 use full attention**. The other
24 use Gated DeltaNet. They maintain recurrent state instead of the same
growing per-token K/V arrays.

Its full-attention layers have 4 KV heads, each with dimension 256:

| Context, one sequence | FP16 K/V | Q8_0 K/V storage |
| --- | ---: | ---: |
| 4,096 | 128 MiB | 68 MiB |
| 8,192 | 256 MiB | 136 MiB |
| 16,384 | 512 MiB | 272 MiB |
| 32,768 | 1,024 MiB | 544 MiB |

These columns count only the eight full-attention layers. Recurrent state,
checkpointing, and runtime buffers must be added separately. The Q8 column is
storage arithmetic, not confirmation that a particular backend supports it
for this hybrid model.

At our 4K setting, switching those arrays from FP16 to Q8 would save **60 MiB**.
That is a useful result: cache precision is probably not where we should start
investigating a 102-second first response.

Context and concurrency can change that decision. Two independent 8K sequences
need twice the conventional cache of one. A long conversation also brings
more history to process and read.

The context budget includes system instructions, template tokens, conversation
history, the new prompt, generated reasoning, and the answer. A 3,000-token prompt
plus a 1,500-token answer exceeds a 4,096-token budget. Check the runner's
truncation behavior before assuming it retained the whole conversation.

Run lesson 2 with different `--context` and `--sequences` values. It uses the
same storage calculations as this section.

## 6. What lower precision buys

Suppose a block uses a scale of 0.1. The value 0.37 might be stored as integer
4 and reconstructed as 0.4. Less storage, less precision.

Real formats add choices about block size, scales, outliers, and which tensors
use which precision. `Q4_K_M` is a weight-quantization recipe, not a statement
that every stored value occupies four bits. FP16 and BF16 both occupy two bytes,
but allocate their bits differently between range and precision.

Our baseline combines **Q4_K_M weights and FP16 cache**. Those settings control
different data. You can change one without changing the other.

Lower-precision weights can reduce memory traffic during decode. The kernel
still has to unpack the representation and perform the arithmetic, so the
benefit depends on the implementation. For the cache, compare saved storage
with the amount of cache the architecture has in the first place.

Quality needs a separate measurement. A small numerical change can alter the
chosen token, which changes the prefix for the next step. Conversely, a different
sentence can be equally correct. Vector similarity alone does not settle this.

Lesson 3 compares attention outputs for synthetic queries. Its Q8-like encoding
illustrates block scaling; its toy4 encoding is explicitly not llama.cpp's
packed Q4_0. Use it to understand rounding, then evaluate actual answers before
choosing a format.

For our coding prompt, a useful rubric would include an empty string, punctuation,
mixed case, and a negative case such as `"hello"`. The generated answer supplied
three positive assertions. That is not enough to establish general correctness.

## 7. FlashAttention removes a temporary matrix

Full prefill attention can materialize an `N x N` score matrix per head.
At N=8,192, one FP32 matrix occupies 256 MiB. This is temporary workspace,
separate from the persistent K/V arrays.

FlashAttention processes tiles and maintains a running softmax, avoiding storage
of the complete score matrix. Its GPU implementation combines that algorithm
with on-chip memory and fused kernels.

The running state for one query is:

```text
m = largest score seen
l = sum of exp(score - m)
o = sum of exp(score - m) * value
```

If a new tile raises the maximum, rescale the old accumulators:

```text
m_new = max(m, largest score in new tile)
correction = exp(m - m_new)
l_new = correction * l + sum(exp(new_scores - m_new))
o_new = correction * o + sum(exp(new_scores - m_new) * new_values)
output = o_new / l_new
```

For example, the first tile contains score 2 with value 10. Its state is
`m=2, l=1, o=10`. The next tile contains score 4 with value 20:

```text
correction = exp(2 - 4) = approximately 0.1353
l_new = 0.1353 + 1
o_new = 0.1353 * 10 + 20
output = approximately 18.808
```

This matches weighting the two values with `softmax([2, 4])`.
Lesson 5 implements the single-query reduction in NumPy. It checks the math;
it does not reproduce a GPU FlashAttention kernel.

The full score matrix disappears, but the cache and other state remain.
The tile itself is also only part of a kernel's workspace: it needs Q/K/V
data, accumulators, and enough active work to occupy the GPU.

Ollama documents quantized KV cache alongside FlashAttention. Support differs
by architecture and backend; check the effective settings in the installed
runner's logs. Current llama.cpp can auto-enable required FlashAttention for
quantized V cache or report an unsupported configuration.

## 8. Investigate decode after startup

The longer baseline answers generated at 10.25 and 14.15 tokens/s. Ollama
reported full GPU placement, so unexpected CPU layer placement was not apparent
in that run. Placement does not tell us whether the kernels used the GPU well.

For one sequence, decode often spends substantial time fetching weights.
Prefill can reuse weights across many token positions; decode has less work
to amortize each read.

A deliberately narrow bandwidth bound is:

```text
tokens/s <= available bytes/s / weight bytes read per token
```

Assume 3 GiB of weight traffic per token and 100 decimal GB/s of bandwidth:
the result is about 31 tokens/s. These are example inputs, not measured laptop
specifications. The bound omits cache reads, compute, recurrent updates,
unpacking, host work, and kernel launches.

Use lesson 7 to vary the inputs. Treat the result as a way to explain a possible
limit, not a target the hardware must achieve.

For MoE models, expert selection changes how much weight data is used per token.
All required experts still need to be stored or fetched. Active parameter count
describes only part of the cost. Similarly, CPU offloading is not a fixed slowdown:
its effect depends on how much work is moved and where transfers occur.

### Choose the next measurement

| Observed problem | Next experiment |
| --- | --- |
| Slow loading, fast later requests | Compare loaded and unloaded requests; record load separately |
| Slow processing of long prompts | Hold context capacity fixed and vary actual input length |
| Slow long answers | Compare decode rates on longer fixed tasks and inspect GPU conditions |
| Memory pressure | Reduce context or simultaneous requests before changing precision |
| Many users waiting | Measure queue delay and completion rate under controlled arrivals |

A shorter answer is also an optimization when it still does the job. Reducing
240 output tokens to 120 at 12 tokens/s saves roughly ten seconds. That can
matter more to the user than a small kernel improvement.

## 9. Make comparisons you can trust

Change one variable, retain the original configuration, and use a fixed task
suite. Record the model digest as well as its tag: a tag can point to a newer
artifact later.

Warm-up and measurement should be separate. Repeat each condition several times
and report the median and spread. Alternate A/B order if clocks, temperature,
or background activity might drift. Four requests do not support a useful p95.

There are two different kinds of reuse:

**Loaded weights:** the model remains in memory, avoiding another load.
**Cached prefix:** an already-processed prefix can avoid some prefill work.

Repeating the exact prompt may get both. Unloading the runner does not clear
Windows' filesystem cache, so it also does not reproduce a reboot-cold start.
Label the conditions you created rather than calling every first request "cold."

Keep input length separate from configured capacity. Reserving 8K context and
sending a 20-token question measures neither 8K prefill nor long-context retrieval.
The input-length lab supplies documents with known facts and records the runtime's
actual prompt token count.

Read the outputs. Our explanation request asked for **under 120 words**; the
answer contained exactly 120 whitespace-separated words. It also described
architecture change as a requirement of distillation. Those are failures a
tokens-per-second table would miss.

## 10. What changes when two people use the server?

With one slot, requests wait while the active request finishes. Consider a
simple example: every request occupies the runner for ten seconds, and requests
arrive at seconds 0, 4, and 8.

| Request | Arrives | Starts | Finishes | Queue wait | Total latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0 s | 0 s | 10 s | 0 s | 10 s |
| B | 4 s | 10 s | 20 s | 6 s | 16 s |
| C | 8 s | 20 s | 30 s | 12 s | 22 s |

The model did not get slower. Waiting made the service slower. If arrivals
continue every four seconds, a larger queue only postpones rejection.

Batching can share weight reads across sequences. Continuous batching lets
finished requests leave and new requests enter the active batch. This may
improve total throughput while reducing each user's token rate.

Suppose one request gets 12 tokens/s; with two active, each gets 8. The service
now produces 16 tokens/s in total, but each individual answer takes longer.
These numbers illustrate the tradeoff. More slots also require more state and
can cause offloading when memory is tight.

That is why the baseline uses one parallel request. Increase it only with a
workload that records both per-request latency and total completions, including
queue waits and errors. CPU thread count is not a suitable default for GPU slots.

## 11. Keep the server configuration understandable

The server process owns its model store and runtime settings. The CLI and our
Python client send HTTP requests to it. Each server uses its own configured
model directory, even if both servers are running on the same laptop.

The original deployment listens on port 11435. An existing Ollama installation
may use port 11434. [SETUP.md](SETUP.md) shows how to identify the endpoint and
use the installation you already have.

In PowerShell, `$env:NAME = 'value'` affects that process and its children.
It does not modify an already-running server. `setx` changes future processes
and persists beyond the experiment, which makes it less convenient for A/B runs.

A launcher can override inherited settings. The original `Start-Server.ps1`
explicitly sets FP16 cache and one request slot. To change those settings, edit
the intended launcher and restart that server; an assignment in a client terminal
will have no effect.

Keep-alive is about model residency, not the lifetime of the HTTP process.
The server can remain available after unloading the model. Conversely, a server
started inside a temporary terminal session can stop when that session closes.

For a personal laptop, bind to loopback. Before serving other users, add
authentication, TLS, request limits, timeouts, and overload handling in front of
the inference API. CORS does not provide authentication. Treat prompt logs as
private data, and treat generated code as untrusted text rather than an instruction
to run it.

For this deployment, start with the load/prefill split. Once repeated runs explain
the slow start, use longer fixed answers to study decode. The calculations above
give a concrete reason to leave cache quantization until memory is actually the
constraint.
