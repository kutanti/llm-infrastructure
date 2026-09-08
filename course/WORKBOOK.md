# Infrastructure workbook

The small experiments run from the repository root with NumPy.
The larger capstones specify evidence to produce on your chosen training or
serving stack. They are not claims that the course has already run those workloads.

## Lab 1: distinguish capacity from traffic

You store 3 GiB of weights and assume they are read once per generated token.
For an illustrative bandwidth of 100 decimal GB/s, compute the weight-only
throughput bound.

```powershell
python 07_performance_budget.py --weight-gib 3 --bandwidth-gbs 100
```

Now list operations omitted from this estimate. Would reducing the weight size
by half necessarily double observed throughput?

<details>
<summary>Answer</summary>

`100e9 / (3 * 1024^3)` is about 31 tokens/s. It omits compute, cache reads,
recurrent updates, kernel overhead, host work, and transfers. Halving one
component of the traffic cannot guarantee doubling whole-request performance.

</details>

## Lab 2: budget a training step

Use the example configuration from the training chapter:
BF16 weights and gradients, an FP32 master copy, and two FP32 Adam moments.

Calculate parameter-related storage for a one-billion-parameter model. Then
explain what reducing the microbatch can and cannot save.

<details>
<summary>Answer</summary>

`(2 + 2 + 4 + 8) * 1e9 = 16e9` bytes, about 14.9 GiB.
This excludes activations and workspace. A smaller microbatch reduces some
activation storage, not the unsharded parameter or optimizer-state storage.
An actual implementation can use a different precision/copy arrangement.

</details>

## Lab 3: train and merge an adapter

```powershell
python -m course.demos.lora
```

Find the code that freezes the base, initializes B to zero, computes dA and dB,
and merges the adapter.

Why is dA initially zero while dB need not be? What happens if both A and B
start at zero?

<details>
<summary>Answer</summary>

The derivative for A contains B. With B=0 it vanishes initially, while B's
derivative uses the nonzero A. If both are zero, both derivatives vanish in
this parameterization, so the adapter cannot begin learning.

</details>

The lab's target update is rank two and the adapter rank is two. As an extension,
change the adapter to rank one while keeping the original target-update
construction rank two. Predict whether exact fitting remains possible.
Retain the original version to compare held-out error.

The existing assertion expects the original solvable setup; a failed convergence
assertion after your edit can be evidence of the new capacity limit.

## Lab 4: inspect the distillation gradient

```powershell
python -m course.demos.distillation
```

Read `course\demos\objectives.py`. Derive why the derivative of the T-squared
soft cross-entropy contains T rather than T-squared.

Compare held-out teacher agreement with KL divergence. Which can improve even
when the top-ranked class does not change?

<details>
<summary>Answer</summary>

Differentiating the student's scaled logits contributes `1/T`. Multiplication
by `T^2` leaves `T * (p_student - p_teacher)` before batch normalization.

KL measures the probability distribution, not just its winner, so it can
improve while agreement stays unchanged.

</details>

The teacher has six input features; the student sees four. The synthetic labels
do not provide independent real-world correctness. Design one additional metric
you would need before deploying a distilled support-ticket classifier.

## Lab 5: follow an outlier through quantization

```powershell
python -m course.demos.quantization
```

Compare tensor-wide and 32-value-group scales. Identify the exact arrays counted
by `nbytes` and the precision used for scales.

Change the outlier from 10 to 1 and rerun. Then restore it and remove the input
rescaling that reduces the outlier channel's activity. Explain why weight error
and layer-output error need not move together.

<details>
<summary>Answer</summary>

An input matrix weights different channels differently. An error in a rarely
active channel can have little output effect, while many small errors in active
channels accumulate. Grouping changes which elements share an outlier's scale.
The lab counts actual INT8 codes plus FP32 scales, not a GGUF Q8_0 encoding.

</details>

## Lab 6: budget concurrent cache allocations

```powershell
python 02_kv_cache_size.py --context 8192 --sequences 1
python 02_kv_cache_size.py --context 8192 --sequences 4
```

Choose one conventional model row. Calculate how much is saved by Q8 relative
to FP16 and how much is added by four independent sequences.

Now assume two requests share their first 4K tokens. Which part may be shared,
and what must happen when their generations diverge?

<details>
<summary>Answer</summary>

Compatible retained prefix state can be referenced by both requests.
Each divergent suffix needs its own state. Allocation granularity and partial
shared blocks can add copy-on-write overhead. Shared model weights do not imply
shared cache for unrelated prefixes.

</details>

For live measurements, use the input-length exercises in [LABS.md](../LABS.md).
Keep configured capacity and actual input length as separate variables.

## Capstone A: a repeatable single-GPU inference experiment

**Question:** which configuration meets a stated task-quality and response-time
target on your hardware?

Use an already-supported model, one server, and a fixed prompt suite. Include
short interactive requests, long-input retrieval, and longer generation.
Define correctness criteria before running.

Record first output, first answer, prefill and decode durations, generated tokens,
completion reason, and your GPU memory measurement method. Add repeated runs and
document prefix reuse. Change one variable: context capacity, cache precision,
weight artifact, or request concurrency.

**Submit:**

```text
model/runtime/configuration manifest
prompt suite and correctness rubric
raw outputs and timing records
memory observations with timestamps
comparison table including failures and spread
decision and rollback configuration
```

Explain whether the result is limited by memory capacity, bandwidth, compute,
startup, queueing, or a still-unmeasured mechanism.

## Capstone B: a small GPU SFT/LoRA job

**Question:** does adaptation improve a repeated task beyond a fixed prompting
baseline without unacceptable regressions?

Use the training chapter's protocol. Choose a small text checkpoint supported
by your GPU training stack. On a 6 GiB device, do not start by assuming a
multimodal 4B inference model will also fit for training.

The dataset must have consistent labels and a meaningful held-out split.
Inspect template rendering, loss masks, sequence truncation, and trainable
parameters. Overfit a tiny subset before committing to the full run.

**Submit:**

```text
dataset schema, split policy, and representative non-private examples
base-model baseline and scoring code or rubric
library/checkpoint revisions and effective training settings
one-step memory/time breakdown
learning curves and held-out task results
adapter artifact and its exact base dependency
merged/exported artifact comparison if applicable
```

Make one controlled memory tradeoff, such as shorter sequence length or gradient
checkpointing. Explain both its memory saving and its effect on work or task coverage.

## Capstone C: distill, then quantize

**Question:** can a smaller deployable student meet the task requirement?

Build a teacher prompt set, generate or collect permitted teacher targets, and
record filtering. Keep independent correctness labels or a separate evaluation
procedure. Compare the base student with the trained student.

Only after measuring the training change, quantize the resulting student using
a supported recipe and evaluate again. This produces three distinct artifacts:

```text
base student
adapted/distilled student
quantized adapted/distilled student
```

**Submit:** teacher identity and generation settings, data accounting, student
objective, validation results, final-artifact quality, and matched serving
measurements. Report teacher agreement separately from task correctness.

A distillation or quantization step that fails the task target is a result,
not a reason to omit that configuration from the comparison.

## Capstone D: capacity under load

This needs a serving stack that supports the concurrency being studied.

Generate a controlled arrival pattern with bounded load. Compare one request
slot with a larger supported configuration. Measure per-request queue wait,
first answer, completion latency, successful requests/second, total generated
tokens/second, errors, and memory.

A fixed-concurrency client stops submitting when requests slow down, which can
hide an overloaded arrival rate. State whether the load generator is closed-loop
(fixed outstanding requests) or open-loop (scheduled arrivals).

**Submit:** workload mix, arrival policy, latency distribution with enough samples,
overload behavior, and a chosen admission limit. Describe how cancellation frees
resources and how private prompts are kept out of shared reports.

Do not expose an unauthenticated laptop endpoint publicly to produce this test.
Local requests are sufficient for the initial scheduling experiment.
