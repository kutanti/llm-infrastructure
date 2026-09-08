# 6. Serving and distributed systems: turn a model into a bounded service

A working model is not yet a reliable service. Requests arrive unpredictably,
vary in length, compete for memory, and disappear halfway through generation.
Serving shares resources while meeting explicit latency and correctness requirements.

Read [GPU systems](01-gpu-systems.md) and
[KV cache](05-kv-cache-and-quantization.md) first.

## 6.1 Trace one request and name the clocks

A typical request passes through:

```text
receive -> authenticate/validate -> tokenize -> admission/queue
        -> prefill -> sample first token -> decode repeatedly -> finish
```

Delivery can overlap generation. Model/adapter loading, compilation, and
connection establishment add cold-path work.

**Prefill** processes prompt positions and creates their KV state. Its final
logits can produce the first output token. Subsequent **decode** iterations
consume generated tokens, extend the cache, and predict further tokens.

Record these metrics with explicit boundaries:

- **TTFT**, time to first token: client submission to first generated token
  received. It includes queueing and delivery if measured at the client.
- **Inter-token latency**: time between successive output tokens. Report its
  distribution when interruptions matter.
- **TPOT**, time per output token after the first: commonly
  `(last-token time - first-token time) / (output tokens - 1)`.
  It is undefined for a one-token response.
- **End-to-end latency**: submission to completion, including final protocol
  overhead if that is the chosen boundary.
- **Throughput**: completed requests/second or output tokens/second over a
  declared interval. Count input tokens separately.

Suppose admission and queueing take 120 ms, tokenization 10 ms, prefill and
first-token sampling 70 ms, and first delivery 10 ms. TTFT is 210 ms. With
101 output tokens and a further 3.0 seconds of streaming, TPOT is 30 ms and
last-token latency is 3.21 seconds. Protocol completion may be later.

Metadata is not a generated token. Distinguish first generated reasoning
from first user-visible answer, and server timing from client latency.

## 6.2 Fixed batching, continuous batching, and chunked prefill

In a **fixed batch**, several requests start together. Without dynamic
replacement, a finished request leaves an unused slot or masked work until
the longest request completes. For output lengths 10, 20, 30, and 100,
there are 160 useful request-token steps over a possible `4 * 100 = 400`
slots: 40% useful slot occupancy under that simple model.

**Continuous batching** revisits membership at iteration boundaries. Finished
requests leave; new requests enter when their prefill and resource requirements
can be accommodated. A decode iteration might process one token each for
32 active requests. It is not generating 32 successive tokens of one request
in parallel: those tokens depend on earlier samples.

This shares weight reads and avoids waiting for the longest member.
Positions, cache blocks, sampling state, and cancellations remain request-specific.

Prefill complicates fairness. One large prompt can occupy the GPU long enough
to interrupt all existing streams. **Chunked prefill** splits prompt processing
into token ranges and schedules those ranges alongside decode work. Each
chunk attends to the appropriate earlier context; the service still needs
the full prompt processed before producing that request's first token.

Smaller chunks offer more opportunities to resume decode but can increase
launch overhead and reduce matrix efficiency. Larger chunks improve prefill
efficiency but may worsen inter-token latency. A chunk's token count is not
a fixed execution time: context length, shapes, and implementation matter.

## 6.3 Admission control needs both bytes and time

Use a dense-attention example with 32 layers, 8 KV heads, head dimension 128,
and two bytes per cached element:

```text
KV bytes/token = 2 (K and V) * 32 * 8 * 128 * 2
               = 131072 bytes = 128 KiB
```

Suppose measurements show an 8 GiB KV budget **after** weights, workspace,
runtime overhead, and a safety reserve. That stores 65,536 token positions
under this simplified model.

With 16-token pages, each page costs 2 MiB. A request with a 3,000-token prompt
and a strict maximum of 1,000 output tokens conservatively reserves
`ceil(4000 / 16) = 250` pages, or 500 MiB. Sixteen such requests reserve
8,000 MiB; a seventeenth would exceed the 8,192 MiB budget.

This **memory ceiling** assumes no prefix sharing, a uniform cache layout,
and separately budgeted page tables. Reserving even the final output token,
which need not enter the cache, is conservative.

Suppose a hypothetical load test gives these steady decode results for the
intended context-length distribution:

| Active requests | Example p95 inter-token latency | Satisfies 40 ms target? |
| ---: | ---: | --- |
| 8 | 24 ms | Yes |
| 12 | 33 ms | Yes |
| 16 | 48 ms | No |

Ignoring prefill interference, the service can support twelve under that
tested target, even though sixteen fit. Mixed prefill/decode traffic must be
tested before treating twelve as a production limit.

For a 500 ms TTFT objective with 180 ms budgeted outside the queue, waiting
must fit 320 ms. Otherwise reject, defer, or route elsewhere. Estimate waiting
from observed service behavior: a 200-token prompt and a 20,000-token prompt
are not equal jobs.

Reserving each request's maximum growth is predictable but conservative.
Allocating incrementally improves utilization but needs a policy for exhaustion:
preemption, recomputation, swapping where supported, or rejection before
admission. "Allocate until out of memory" is not a policy.

Paged KV storage reduces fragmentation and avoids contiguous maximum-size
allocations, not the bytes needed for actual states. Prefix sharing needs
compatibility and ownership rules; see [Chapter 5](05-kv-cache-and-quantization.md).

## 6.4 Queueing and the concurrency trap

Concurrency is the number of requests in progress, not an intrinsic GPU speed
setting. Increasing it can raise aggregate throughput while worsening each
request's TPOT and TTFT.

Little's law relates long-run averages in a stable system:

```text
average requests in system = arrival rate * average time in system
```

At 4 requests/second and 2 seconds mean residence time, the average is eight
requests, including those queued. This is not a proof that eight active GPU
slots are sufficient, nor a p95 guarantee.

As offered work approaches service capacity, bursts can produce large queues.
Bound queue length and waiting time; define overload responses and client
retry behavior. Immediate unlimited retries amplify overload. Fairness may
need per-user limits or aging so a stream of short requests does not starve
long ones. Deadlines and output limits make resource commitments finite.

## 6.5 Parallelism means deciding what to replicate and communicate

Training and serving parallelism solve different problems.

### Data parallelism and DDP

In inference **data parallelism**, each replica holds a full model and handles
different requests. There is ordinarily no gradient synchronization. This
raises fleet throughput and isolates queues, but does not make a too-large
model fit on one replica.

During training, **DistributedDataParallel**, or DDP, gives replicas different
training examples and synchronizes gradients, usually through bucketed
all-reduces. Each replica still holds model and optimizer state unless another
technique shards them. Communication can overlap backward computation, but
small microbatches leave less compute to hide it.

### ZeRO and FSDP

ZeRO reduces replicated training state: stage 1 shards optimizer state,
stage 2 also shards gradients, and stage 3 also shards parameters. FSDP
similarly offers fully sharded training configurations, gathering parameters
for computation and resharding as configured.

The capacity saving exchanges persistent replication for communication and
transient memory. Peak memory still includes materialized parameter groups,
activations, and buffers; dividing the single-device total by device count
is not a sufficient estimate. Offload adds host-memory and interconnect costs.

### Tensor parallelism

**TP** partitions tensor operations inside a layer. For `Y = XW`, splitting
`W` by output columns produces different output features on each device;
splitting by input rows produces partial sums that must be combined. Practical
transformers pair partitioned projections and use collectives at selected
points rather than always gathering every intermediate.

TP can make a layer fit and parallelize its compute. However, activation
communication occurs repeatedly across layers. Latency-sensitive decode with
small matrices can spend more time coordinating than it saves computing.

### Pipeline parallelism

**PP** puts consecutive groups of layers on different devices. Activations
cross stage boundaries. Independent microbatches can occupy different stages,
but a single token must still traverse the stages in order.

In a simplified forward-only pipeline with `p` equal stages and `m`
microbatches, utilization is approximately `m / (m + p - 1)`. Four stages
and four microbatches give `4/7`, about 57%, before communication or imbalance.
Training schedules have additional forward/backward dependencies. Autoregressive
serving also has feedback between consecutive tokens.

### Expert parallelism

In mixture-of-experts models, **EP** places different experts on different
devices. A router selects experts for each token, tokens are dispatched to
those devices, and expert outputs return for combination. This commonly
requires all-to-all communication. Uneven routing creates load imbalance;
expert capacity and token-handling policies affect both quality and speed.
Sparse expert activation does not mean expert weights require no storage.

Methods can combine: replicas may each use TP and PP internally.
Describe device groups, not just an "eight-GPU deployment."

## 6.6 Accumulation is not a serving slot

Training gradient accumulation processes several microbatches before an
optimizer update. With microbatch size 2, accumulation 8, and four
data-parallel replicas, the effective batch is:

```text
2 * 8 * 4 = 64 examples/update
```

Also record nonpadding tokens/update for variable-length text. This does not
imply 64 simultaneous activation sets or generation slots. Serving retains
independent KV state; accumulation combines gradients across training work.

## 6.7 Interconnects and disaggregation

PCIe topology, NVLink availability, peer-to-peer support, host staging, and
network fabric determine communication performance. Consumer devices may lack
assumed datacenter connectivity. Verify the transfer path.

For one illustrative collective payload, moving 64 MiB over a sustained
20 GB/s path takes at least about 3.36 ms before startup and protocol overhead.
If a shard's compute takes 1 ms, communication is already consequential.
An all-reduce's actual traffic depends on its algorithm and device count;
do not substitute one payload transfer for a complete collective model.

**Prefill/decode disaggregation** assigns the two phases to separate workers.
It can isolate decode latency from prefill bursts and let each pool use
different batching or hardware. But KV state must move, or be remotely
accessible, and the receiving representation must be compatible.

For the earlier cache geometry, an 8,192-token prompt produces 1 GiB of KV.
A 25 GB/s effective transfer path needs at least about 43 ms to move it once.
Transfer coordination, serialization, contention, and destination allocation
add cost; pipelining may hide part of it. Compare that cost against the
scheduling benefit. Disaggregation can lose on short prompts or weak links.

Two GPUs need not halve latency: replication improves aggregate throughput,
sharding adds synchronization, pipelines have dependencies, and offloading
can become transfer-bound.

## 6.8 Measure a service, not a favorable demonstration

Start with a reproducibility record: checkpoint revision, tokenizer and chat
template, adapters, weight and KV formats, runtime version, hardware topology,
context/output limits, sampling settings, prefix-cache state, and workload
length distributions. Fixed seeds alone do not ensure identical results across
backends or batching arrangements.

Use a code-independent lab: a spreadsheet and existing request logs suffice.
Construct a timeline of arrivals, admissions, first tokens, completions, and
cancellations. Calculate TTFT and TPOT per request, then median and p95. Record
sample count and failed requests; dropping failures makes overload look better.

Compare sequential requests against an arrival schedule independent of
completions. A closed-loop client that waits before submitting more work can
hide overload. Separate offered load from achieved throughput. Load-test shared
services only with authorization.

Observe queue age, active sequences, cached tokens, page availability, prefill
and decode time, errors, and memory peaks. Separate cold model/adapter loads
from warm service and measure prefix-cache hits rather than assuming them.
Report p95 TTFT alongside p95 TPOT or inter-token latency: an average TPOT can
hide a long midstream stall.

Cancellation must propagate to scheduling and free request-owned cache,
while respecting shared-prefix references and in-flight work. Otherwise
disconnected clients consume capacity invisibly. Bound generated output,
handle deadlines, and shed excess load deliberately.

## 6.9 Deployment boundaries that preserve the experiment

Default a local experimental endpoint to loopback, not every network interface.
Remote deployment needs authentication, authorization, and TLS directly or
through a trusted proxy, with no bypassing unprotected backend port.

Prompts, retrieved documents, outputs, and logs can contain sensitive data.
Limit who can read them, avoid logging raw content by default, and define
retention. Record metrics without making the monitoring system a prompt archive.

Allowlist validated adapters and model artifacts. Bind requests to the intended
tenant and revision; include model/adapter compatibility in prefix-cache identity.
Isolate cache domains when shared-prefix timing or data isolation matters.
Matching token IDs alone do not make cached states compatible.

Readiness means the intended model can accept work, not merely an open port.
Use bounded startup, health checks, graceful draining, rollback, and explicit
versioning to preserve both reliability and repeatability.

## Exercises

<details>
<summary>1. With the chapter's 128 KiB/token cache and 16-token pages, what does a 4,001-token reservation cost?</summary>

It needs `ceil(4001/16) = 251` pages, each 2 MiB: 502 MiB. Page rounding
reserves 4,016 positions, leaving 15 unused positions. Sixteen such reservations
cost 8,032 MiB, still below 8,192 MiB, but the measured latency limit remains
separate.

</details>

<details>
<summary>2. Throughput rises from 250 to 350 output tokens/s while p95 TTFT doubles. Is the second configuration better?</summary>

Only relative to a stated objective. It may violate an interactive latency
SLO despite better aggregate efficiency. Compare the same arrival process,
length distribution, errors, and quality; examine queue time and decode
latency separately.

</details>

<details>
<summary>3. A model fits one GPU. Should a second GPU use TP or an independent replica?</summary>

For many independent requests, a replica avoids per-layer communication and
is a strong baseline. TP may help an individual request if compute savings
outweigh communication, or meet another capacity constraint such as cache.
Benchmark both against the actual SLO and topology; GPU count alone cannot
choose the winner.

</details>

## Primary references

- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180):
  cache paging, sharing, and serving measurements.
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu):
  iteration-level scheduling.
- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal):
  chunked prefill and throughput/latency scheduling tradeoffs.
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
  and [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html):
  implementation semantics and memory/communication controls.
- [ZeRO](https://arxiv.org/abs/1910.02054) and
  [Megatron-LM](https://arxiv.org/abs/1909.08053):
  optimizer-state partitioning and tensor model parallelism.
- [GPipe](https://arxiv.org/abs/1811.06965) and
  [GShard](https://arxiv.org/abs/2006.16668):
  pipeline microbatching and distributed conditional computation.
- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong):
  prefill/decode disaggregation and goodput under latency constraints.

Return to the [course index](README.md) and
[workbook](WORKBOOK.md) to assemble an end-to-end experiment.
