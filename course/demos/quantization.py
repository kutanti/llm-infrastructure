"""Inspect actual INT8 code/scale bytes and errors for two grouping choices."""

import numpy as np


def quantize_groups(values, group_size):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or group_size < 1 or values.size % group_size:
        raise ValueError("group_size must divide a nonempty tensor")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    groups = values.reshape(-1, group_size)
    scales = np.abs(groups).max(axis=1, keepdims=True) / 127
    normalized = np.divide(groups, scales, out=np.zeros_like(groups), where=scales != 0)
    codes = np.rint(normalized).clip(-127, 127).astype(np.int8)
    return codes, scales


def reconstruct(codes, scales, shape):
    return (codes.astype(np.float32) * scales).reshape(shape)


def relative_error(reference, actual):
    norm = np.linalg.norm(reference)
    if norm == 0:
        raise ValueError("relative error is undefined for a zero reference")
    return float(np.linalg.norm(actual - reference) / norm)


def main():
    rng = np.random.default_rng(23)
    weights = rng.normal(0, 0.05, (64, 64)).astype(np.float32)
    weights[0, 0] = 10
    inputs = rng.normal(size=(16, 64)).astype(np.float32)
    # Reduce the outlier channel's activity to expose why weights alone are insufficient.
    inputs[:, 0] *= 0.01
    reference = inputs @ weights
    print("Synthetic weights: mostly small values, one large outlier.")
    print("Codes use int8; scales use FP32. This is not GGUF Q8_0.")
    print(f"Original FP32 matrix: {weights.nbytes} bytes\n")
    print(f"{'scheme':18}{'code+scale bytes':>18}{'weight rel L2':>18}{'output rel L2':>18}")
    for label, size in (("tensor-wide", weights.size), ("groups of 32", 32)):
        codes, scales = quantize_groups(weights, size)
        restored = reconstruct(codes, scales, weights.shape)
        output = inputs @ restored
        print(f"{label:18}{codes.nbytes + scales.nbytes:18}"
              f"{relative_error(weights, restored):18.5f}"
              f"{relative_error(reference, output):18.5f}")
        assert np.isfinite(restored).all()
    zeros = np.zeros((2, 32), dtype=np.float32)
    codes, scales = quantize_groups(zeros, 32)
    np.testing.assert_array_equal(reconstruct(codes, scales, zeros.shape), zeros)
    print("\nMore groups mean more scales, but can isolate outliers.")
    print("We reconstruct the full matrix before multiplying to inspect error.")
    print("This is not a packed/fused kernel and establishes no inference speedup.")


if __name__ == "__main__":
    main()
