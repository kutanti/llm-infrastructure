"""Measure eager projections separately from profiler/compilation overhead."""

import argparse
import json
from pathlib import Path
import time

import torch


def measure(device, rows, width, iterations, output, compiled=False):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment")
    if min(rows, width, iterations) <= 0:
        raise ValueError("Rows, width and iterations must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    x = torch.randn(rows, width, device=device)
    w = torch.randn(width, width, device=device)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    operation = torch.matmul
    compilation_s = None
    if compiled:
        operation = torch.compile(operation)
    with torch.inference_mode():
        if compiled:
            synchronize()
            start = time.perf_counter()
            operation(x, w)
            synchronize()
            compilation_s = time.perf_counter() - start
        for _ in range(5):
            operation(x, w)
        synchronize()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iterations):
            y = operation(x, w)
        synchronize()
        elapsed = time.perf_counter() - start
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=True) as profiler:
            with torch.profiler.record_function("projection"):
                operation(x, w)
            synchronize()
        profiler.export_chrome_trace(str(output / "trace.json"))
    torch.testing.assert_close(y, x @ w)
    flops = 2 * rows * width * width
    minimum_bytes = (rows * width + width * width + rows * width) * x.element_size()
    result = {
        "torch_version": str(torch.__version__), "device": device,
        "shape": {"rows": rows, "width": width}, "dtype": str(x.dtype),
        "iterations": iterations, "seconds_per_projection": elapsed / iterations,
        "flops_per_projection": flops, "minimum_tensor_bytes": minimum_bytes,
        "arithmetic_intensity_flops_per_byte": flops / minimum_bytes,
        "achieved_gflops_per_s": flops * iterations / elapsed / 1e9,
        "compiled": compiled, "compilation_and_first_call_s": compilation_s,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "note": "Timing excludes profiler and initial warmup. Traffic is a tensor-size lower bound, not a hardware counter.",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--compile", action="store_true", help="Requires a supported compiler/toolchain")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(measure(args.device, args.rows, args.width, args.iterations,
                             args.output, args.compile), indent=2))


if __name__ == "__main__":
    main()
