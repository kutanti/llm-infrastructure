# 1. GPU systems: explain the bottleneck before changing the model

The [foundations](../foundations/README.md) followed tensors through a
transformer. Now follow the work through a machine. A useful performance
explanation identifies an operation, its shapes, the bytes it moves, and the
resource limiting it. "The GPU is slow" identifies none of these.

By the end, you should be able to distinguish a capacity failure from a
bandwidth limit, explain why prefill and decode behave differently, and design
a measurement that separates GPU execution from loading and CPU work.

## 1.1 CPU and GPU: different ways to spend silicon

A CPU devotes substantial resources to making a small number of instruction
streams progress quickly: branch prediction, out-of-order execution, and large
caches. This suits tokenization, scheduling, file processing, and irregular
control flow.

A GPU devotes more resources to executing many arithmetic operations
concurrently. It needs enough independent work to hide the delay between
requesting data and receiving it. Moving a small operation to a GPU can lose:
launch and transfer overhead may exceed the saved arithmetic time.

In CUDA terminology:

- A **kernel** is a function launched across a grid of thread blocks.
- A **block** contains threads that can cooperate through shared memory and
  block-level synchronization. An ordinary block executes on one streaming
  multiprocessor, or **SM**.
- Threads execute in **warps**, normally 32 threads on NVIDIA GPUs. Warp
  schedulers select ready warps and issue their instructions.

A block is not an SM, and a CUDA thread is not a CPU operating-system thread.
Several blocks can reside on an SM if their register, shared-memory, and
thread requirements fit. Blocks beyond that capacity wait for resources.
If one warp waits for memory, a scheduler can issue work from another.

Branches whose paths differ within a warp can require executing different
paths with different active lanes. Conversely, neighboring lanes accessing
neighboring addresses can combine accesses into fewer memory transactions.
Thus the arrangement of the work matters, not just its total size.

## 1.2 Memory: capacity and bandwidth are different constraints

| Location | Role | Important limitation |
| --- | --- | --- |
| Registers | Thread-local values and accumulators | Per-SM resource budget; spills can become device-memory traffic |
| Shared memory | Explicitly managed, block-cooperative tiles | Limited per SM; bank conflicts and synchronization |
| Device caches | Reuse between memory operations | Reuse and access patterns determine effectiveness |
| Device memory, GDDR or HBM | Weights, activations, KV cache, workspaces | Capacity and sustained bandwidth |
| Host RAM | CPU data, staging, offloaded tensors | Access crosses an interconnect for a discrete GPU |
| Storage | Checkpoints and datasets | Loading is not GPU arithmetic |

PCIe connects host and device; it is not another large register file. Suppose
an illustrative system sustains 300 GB/s from device memory but only 12 GB/s
for a particular host-to-device transfer. Moving 2 GB costs at least about
6.7 ms in the first case and 167 ms in the second. Actual accesses and
transfers have additional overhead. Use measured bandwidth, not the slot's
advertised maximum.

Units also matter. A laptop reporting 6141 MiB has about 5.997 GiB, or
6.439 billion bytes, of reported dedicated capacity. It does not have all of
that available for model weights. Display allocations, CUDA context,
workspaces, cache, and other processes consume space.

On Windows, Task Manager's **shared GPU memory** is system RAM available
through the operating system's memory-management mechanisms, not additional
on-card VRAM at device-memory bandwidth. WDDM can affect residency and
accounting; behavior depends on the driver and allocation path. Do not assume
an application will transparently spill successfully: it may slow severely
or fail allocation. A capacity estimate should start with dedicated memory.

## 1.3 A roofline calculation for a transformer projection

Consider a linear layer:

```text
X: [M, 4096]
W: [4096, 4096]
Y = XW: [M, 4096]
```

With FP16 inputs and outputs, and counting a multiply-add as two floating-point
operations:

```text
work = 2 * M * 4096 * 4096 FLOPs
minimum tensor traffic = 2 * (M*4096 + 4096*4096 + M*4096) bytes
arithmetic intensity I = work / traffic, in FLOPs/byte
```

This traffic model assumes each input is read once and each output written
once. It omits workspace, rereads, and other layer operations. It is a useful
lower bound for transfers between device memory and the processor, assuming
the weights are not already resident in cache.

For a single decode token, `M = 1`: the operation resembles a matrix-vector
product, or GEMV. Work is 33.55 million FLOPs; traffic is approximately
33.57 MB, overwhelmingly weights. Intensity is about 1 FLOP/byte.

For 128 prompt positions, `M = 128`: this is a matrix-matrix product, or GEMM.
Work is 4.295 billion FLOPs; traffic is 35.65 MB. Intensity is about
120.5 FLOPs/byte because many rows reuse the weights.

Now posit a device with **usable** 300 GB/s memory bandwidth and 30 TFLOP/s
compute throughput for this precision and operation. These are teaching
assumptions, not specifications for the laptop.

```text
roofline throughput <= min(compute ceiling, bandwidth * I)
ideal execution time >= max(work / compute ceiling, bytes / bandwidth)
ridge point = 30e12 / 300e9 = 100 FLOPs/byte
```

The single-token case has a memory lower bound of about 112 microseconds
versus a compute lower bound of 1.12 microseconds. The 128-token case has
about 119 microseconds of memory time versus 143 microseconds of compute
time. The first is strongly bandwidth-limited under these assumptions; the
second crosses the ridge point.

This does not predict end-to-end tokens/second. Real execution includes
attention, normalization, launches, cache effects, imperfect tiling, and
dependencies. At long context, decode must also read substantial KV state.
Nevertheless, the calculation explains why batching independent decode
requests can improve weight reuse, and why prefill can exploit more compute.

## 1.4 Tensor cores: storage precision is not execution precision

Tensor cores perform specialized matrix operations on tiles. Eligible data
types, accumulator types, tile shapes, and performance vary by GPU
architecture. A backend must select a suitable kernel; small or awkward shapes
can underuse hardware or select a different implementation.

There are at least three separate numerical choices:

1. The representation stored in memory.
2. The operands actually consumed by arithmetic instructions.
3. The accumulator and output representations.

FP16 operands may accumulate in FP32. A framework may allow TF32 math for
some FP32 matrix operations, subject to its version and precision settings.
A 4-bit checkpoint may use packed weights that are unpacked or dequantized
into another precision before multiplication. "4-bit model" therefore does
not prove native 4-bit tensor-core execution.

Compression reduces bytes but introduces scale reads and conversion work.
It helps only if the implementation and workload benefit enough to offset
those costs. [Chapter 4](04-weight-and-activation-quantization.md) develops
the numerical and kernel consequences.

## 1.5 Occupancy, kernels, and layout

**Occupancy** measures resident active warps relative to the hardware maximum,
not the fraction of peak arithmetic performance achieved. High occupancy can
help hide latency; it cannot repair uncoalesced accesses or insufficient
memory bandwidth. Low occupancy can be adequate when a kernel has enough
instruction-level parallelism.

For illustration, suppose an SM has 65,536 32-bit registers. A 256-thread block
using 64 registers per thread consumes 16,384 registers, so registers alone
allow at most four such blocks. Other limits may reduce that number. Raising
usage to 96 registers per thread consumes 24,576 per block, allowing at most
two before accounting for allocation granularity. A seemingly useful
optimization can reduce latency hiding.

Transformers also launch many non-matrix kernels: normalization, activations,
rotary position transformations, and sampling. **Fusion** combines operations
so intermediates can stay in registers or shared memory instead of being
written out and reread. It also reduces launches. Excessive fusion can
increase register pressure, so fusion is not automatically better.

A transpose may be a cheap view with different strides, but the next kernel
may require a contiguous tensor and trigger a real copy. Measure where the
copy occurs. Pinned host buffers and asynchronous transfers can sometimes
overlap transfer with computation when hardware, streams, and dependencies
permit; writing `non_blocking=True` does not guarantee overlap.

## 1.6 Driver, CUDA toolkit, and framework runtime

The NVIDIA **driver** lets the operating system and applications communicate
with the GPU. The **CUDA toolkit** supplies development tools such as `nvcc`,
headers, and libraries for building CUDA programs. A CUDA-enabled PyTorch
binary distribution commonly supplies its needed CUDA user-space runtime
libraries directly or through package dependencies.

Consequently, running a supported PyTorch binary normally does not require
installing a matching local toolkit. Building custom CUDA extensions may.
The driver must still be compatible with the framework's runtime and GPU.

The "CUDA Version" displayed by `nvidia-smi` indicates driver-supported CUDA
compatibility, not proof that that toolkit is installed or used by Python.
Record the driver version, framework version, framework CUDA build version,
GPU model, and selected backend separately. Installing several toolkits is
not a diagnosis.

## 1.7 Time asynchronous work correctly

A CPU timer around a GPU call may measure only submission: CUDA work usually
executes asynchronously. For an isolated operation, use this protocol:

1. Construct inputs outside the measured section.
2. Warm up the same shapes and paths, allowing initialization and compilation.
3. Synchronize before starting a CPU timer.
4. Run the operation or a repeated group.
5. Synchronize before stopping the timer.
6. Report repetitions and distribution, not one favorable result.

CUDA events can time GPU work on a stream without including the entire CPU
path. They must bracket the relevant work and be complete before reading
elapsed time. Multiple streams require appropriate dependencies. Per-iteration
synchronization changes overlap, so use it for controlled microbenchmarks,
not indiscriminately inside a production pipeline.

Keep separate experiments for cold startup, warm GPU execution, and full
client latency. Include tokenization and streaming in the last one, not by
accident in the first.

## 1.8 A practical diagnostic lab without new dependencies

Use an existing workload and existing monitoring, or analyze the recorded
[laptop case study](../GUIDE.md). No new GPU framework is required.

These are read-only Windows PowerShell monitoring commands; run them yourself
only if `nvidia-smi` is already available. Stop continuous sampling with Ctrl+C.

```powershell
nvidia-smi
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv -l 1
```

Unsupported fields may report `N/A`. GPU utilization is sampled activity, not
achieved FLOPs; memory utilization is not the percentage of VRAM occupied.
One-second samples miss short kernels. WDDM can also limit per-process
visibility.

Make a worksheet with prompt tokens, output tokens, cold/warm status, wall
time, dedicated memory, and concurrent processes. Change one variable: prompt
length, output length, or concurrent requests. State a hypothesis first:

- Long first request, fast repeats: investigate loading and initialization.
- Low activity with long delays: inspect CPU work, queueing, or transfers.
- High activity with slow decode: inspect bandwidth and KV reads, not just
  "GPU percentage."
- Failure after increasing context: calculate growing state and peak workspace.

If existing profilers are available, use a timeline first, such as Nsight
Systems or the PyTorch profiler. Locate CPU gaps, copies, and dominant kernels.
Then use kernel-level measurements, such as Nsight Compute, to test bandwidth
or occupancy hypotheses. Profiling itself adds overhead.

PyTorch's caching allocator distinguishes **allocated** memory held by tensors
from **reserved** memory managed by the allocator. Reserved but unused blocks
may be reusable. `nvidia-smi` additionally sees context and non-PyTorch
allocations. `empty_cache()` can release unused cached blocks; it cannot free
live tensors or make an oversized live workload fit. Measure peaks, not just
the memory remaining after a request.

## 1.9 Bridge to training

Inference retains weights, runtime workspaces, and request KV state. Training
also needs gradients, optimizer state, and activations saved for backward.
A simple mixed-precision Adam budget can be 16 bytes per parameter: 2 for
weights, 2 for gradients, 4 for master weights, and 8 for two FP32 moments.
Actual implementations differ.

At one billion parameters that example already needs 16 GB before activations.
Activation checkpointing trades recomputation for memory; adapters reduce
trainable state but do not eliminate execution through the frozen backbone.
Continue with [training and fine-tuning](02-training-and-fine-tuning.md).
Neither a parameter count nor a VRAM capacity gives a fixed speed guarantee.

## Exercises

<details>
<summary>1. A projection reads 64 MB and performs 128 MFLOPs. At 400 GB/s and 20 TFLOP/s, what is its ideal lower-bound time?</summary>

Intensity is 2 FLOPs/byte. Memory time is 0.16 ms; compute time is 0.0064 ms.
The roofline lower bound is 0.16 ms, before overhead and extra traffic.
Doubling peak arithmetic throughput alone barely changes this bound.

</details>

<details>
<summary>2. A process has 3.5 GiB allocated and 4.5 GiB reserved. Does emptying its cache free 4.5 GiB for a larger model?</summary>

No. The 3.5 GiB held by live tensors remains. Some of the roughly 1 GiB
difference may be releasable, subject to allocator behavior and fragmentation.
Other process allocations also remain. Check peak live memory and the sizes
of upcoming allocations.

</details>

<details>
<summary>3. A CPU timer reports 0.2 ms for a kernel, but synchronized timing reports 8 ms. Which is the user's latency?</summary>

The first may measure submission; the second includes waiting for execution.
Neither necessarily equals user latency, which also includes queueing,
tokenization, other kernels, and delivery. Define and measure that boundary
separately.

</details>

## Primary references

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/):
  execution model, memory hierarchy, synchronization, and compute capabilities.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/):
  bandwidth, coalescing, occupancy, and measurement.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html):
  asynchronous execution, precision controls, and allocator accounting.
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/):
  field definitions and platform limitations.
- [Roofline model paper](https://doi.org/10.1145/1498765.1498785):
  Williams, Waterman, and Patterson's compute-versus-bandwidth model.
