# Inspect the mechanisms behind the pipeline

## Read a projection trace

Use the optional PyTorch environment:

```powershell
.\pipeline\.venv\Scripts\python.exe -m pipeline.profile --rows 1 --width 512 --output results\pipeline\profile-decode
.\pipeline\.venv\Scripts\python.exe -m pipeline.profile --rows 128 --width 512 --output results\pipeline\profile-prefill
```

The two operations share a `[512, 512]` FP32 weight matrix. The first multiplies
one row; the second reuses the weights across 128 rows. `metrics.json` reports
work, a minimum tensor-traffic calculation, and measured time. It does not turn
the traffic estimate into a measured memory-bandwidth counter.

Each `trace.json` is a Chrome trace. Open it in a local trace viewer; do not
upload traces containing sensitive paths, shapes, or application data to a
third-party service. Find the `projection` region and its matrix-operation
children. On CPU there are no CUDA kernels to inspect.

With a CUDA-enabled environment, repeat with `--device cuda`. Compare host
submission with device execution; the wall timer synchronizes around the whole
measured loop, not every matrix call. Profiler collection happens in a separate
pass, so its overhead is excluded from the timing.

An optional `--compile` path uses `torch.compile` and requires a supported
compiler/toolchain. Record compilation/first-call cost separately. Compare
many identical-shaped iterations; one tiny call is not evidence that compilation
helps a service. This example does not implement a custom Triton kernel.

## Why AdamW needs more than a gradient

For gradient `g_t`, Adam maintains exponential moving estimates:

```text
m_t = beta1 * m_(t-1) + (1-beta1) * g_t
v_t = beta2 * v_(t-1) + (1-beta2) * g_t^2
m_hat = m_t / (1-beta1^t)
v_hat = v_t / (1-beta2^t)
w_next = (1 - learning_rate * weight_decay) * w
         - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

The decoupled decay term distinguishes AdamW from simply adding an L2 penalty
to Adam's gradient. The two moments explain part of optimizer memory. Restoring
weights without moments, the update counter, and scheduler state changes the
subsequent trajectory.

Warmup increases the learning rate over early updates rather than starting at
the maximum. Later decay reduces the step size. Neither repairs mislabeled
targets. Gradient clipping limits a gradient norm before the optimizer update;
it should not hide NaN gradients. Inspect the unscaled gradient when mixed
precision uses loss scaling.

For a failing run, first overfit a few examples. If loss does not fall, inspect
which tokens have labels, whether the adapter receives gradients, and whether
the optimizer owns those parameters. If training improves but validation does
not, inspect split design and overfitting before increasing GPU count.

## Online softmax without retaining all scores

For one query and an allowed prefix, attention is:

```text
o = sum_j exp(s_j) * v_j / sum_j exp(s_j)
s_j = q dot k_j / sqrt(head_dim)
```

Stable computation subtracts a maximum from scores. Across tiles, that maximum
may change. Retain three things: running maximum `m`, denominator `l`, and
unnormalized output vector `a`. When a new tile has scores `s`:

```text
m_new = max(m, max(s))
rescale = exp(m - m_new)
p_tile = exp(s - m_new)
l_new = rescale * l + sum(p_tile)
a_new = rescale * a + p_tile @ V_tile
output = a / l
```

Rescaling **both** old accumulators is essential. Without it, earlier tiles
use a different normalization scale. `pipeline.cache.online_attention`
implements these equations and compares them with full attention.

```powershell
python -m pipeline.cache
```

The example has one decode query and 97 cached positions, processed in tiles
of 16. It does not implement GPU FlashAttention or causal training masks.
For multiple training queries, each needs its own allowed-position mask.
Avoiding the full score matrix does not eliminate K/V storage or output state.

## Make cache ownership visible

The same command runs a page-allocation simulation with two slots per page:

```text
A: [10, 20] [30]
B forks A and shares both pages.
A appends 40.
A: [10, 20] [30, 40]
B: [10, 20] [30]
```

The first page remains shared. The partial second page is copied before A
writes it; otherwise B's prefix would change. Releasing A frees only pages with
no remaining references. With no free page, copy-on-write raises an allocation
error without corrupting either request.

Change the example's maximum page count from three to two and locate that
failure. It is an admission/capacity problem, not a reason to overwrite B's data.
The payload consists of token IDs so ownership is easy to inspect; real K/V
contains arrays per attention layer, head, and position.

## What these exercises do not implement

Custom CUDA/Triton kernels, a production paged-attention backend, speculative
decoding, distributed collectives, preference optimization, and a fault-tolerant
cluster deployment remain separate work. The exercises above provide executable
starting points, not substitutes for those systems.
