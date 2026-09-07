"""Lesson 3: synthetic cache precision experiments, not model quality scores.

Q8-like uses 32-element blocks, an FP16 scale and int8 values. NumPy's
rounding is not promised to match a backend bit-for-bit. Toy4 uses 15 signed
levels stored in int8; it is NOT llama.cpp q4_0 and is NOT packed to 4 bits.
"""

import numpy as np

from cache_math import FORMATS


BLOCK = 32


def quantize(x, levels):
    if levels not in (7, 127):
        raise ValueError("This demo supports levels=7 or levels=127")
    if x.size == 0 or x.size % BLOCK:
        raise ValueError("Input must contain a nonzero multiple of 32 elements")
    if not np.isfinite(x).all():
        raise ValueError("Input must be finite")
    rows = x.astype(np.float32).reshape(-1, BLOCK)
    scales = np.abs(rows).max(axis=1, keepdims=True) / levels
    normalized = np.divide(
        rows, scales, out=np.zeros_like(rows), where=scales != 0
    )
    q = np.rint(normalized).clip(-levels, levels).astype(np.int8)
    stored_scales = scales.astype(np.float16)
    return q, stored_scales


def dequantize(q, scales, shape):
    return (q.astype(np.float32) * scales.astype(np.float32)).reshape(shape)


def rounded(x, levels):
    q, scales = quantize(x, levels)
    return dequantize(q, scales, x.shape)


def attention(q, keys, values):
    scores = (keys @ q) / np.sqrt(q.size)
    weights = np.exp(scores - scores.max())
    return (weights / weights.sum()) @ values


def main():
    zero = np.zeros((1, BLOCK), dtype=np.float32)
    np.testing.assert_array_equal(rounded(zero, 127), zero)
    np.testing.assert_array_equal(rounded(zero, 7), zero)
    rng = np.random.default_rng(42)
    print("Attention-output relative L2 error across 10 synthetic queries.")
    print("Different query scales change attention concentration.")
    print(f"{'query scale':>12}{'FP16':>13}{'Q8-like':>13}{'K8/toyV4':>13}{'toy4':>13}")
    for query_scale in (0.25, 1.0, 4.0):
        errors = []
        for _ in range(10):
            keys = rng.normal(size=(512, 128)).astype(np.float32)
            values = rng.normal(size=(512, 128)).astype(np.float32)
            q = rng.normal(size=128).astype(np.float32) * query_scale
            ref = attention(q, keys, values)
            k8, v8 = rounded(keys, 127), rounded(values, 127)
            k4, v4 = rounded(keys, 7), rounded(values, 7)
            outputs = (
                attention(q, keys.astype(np.float16).astype(np.float32),
                          values.astype(np.float16).astype(np.float32)),
                attention(q, k8, v8),
                attention(q, k8, v4),
                attention(q, k4, v4),
            )
            errors.append([
                np.linalg.norm(out - ref) / np.linalg.norm(ref)
                for out in outputs
            ])
        medians = np.median(errors, axis=0)
        assert np.isfinite(medians).all()
        print(f"{query_scale:12.2f}" + "".join(f"{e:13.5f}" for e in medians))

    print("\nReal storage formats (not the toy4 arrays in this program):")
    for name, (block, size) in FORMATS.items():
        print(f"  {name}: {size / block:.4f} bytes/value including block scales")
    print("\nK errors can change which tokens receive attention.")
    print("V errors change the content being blended. Neither is universally")
    print("more harmful. The softmax does not guarantee harmless noise.")
    print("One attention layer cannot predict end-to-end answer accuracy.")
    print("Evaluate real tasks and long contexts before choosing a cache format.")
    print("Neither Q8 nor Q4 is always free, always faster, or always bad.")


if __name__ == "__main__":
    main()
