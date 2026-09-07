"""Lesson 4: different questions can use the SAME keys and values."""

import numpy as np

from cache_math import cache_bytes


def attend(query, keys, values):
    scores = keys @ query / np.sqrt(query.size)
    p = np.exp(scores - scores.max())
    weights = p / p.sum()
    return weights, weights @ values


def main():
    # Toy book labels: first coordinate is about cats, second about code.
    keys = np.array([[2.0, 0.0], [0.0, 2.0]])
    values = np.array([[10.0, 0.0], [0.0, 20.0]])
    queries = np.array([[3.0, 0.0], [0.0, 3.0]])
    outputs = []
    print("Two query heads share exactly one K/V head:")
    for label, query in zip(("cats?", "code?"), queries):
        weights, output = attend(query, keys, values)
        np.testing.assert_allclose(weights.sum(), 1.0)
        outputs.append(output)
        print(f"  {label:6} weights={weights.round(4)} output={output.round(4)}")
    assert not np.allclose(outputs[0], outputs[1])
    print("Shared shelves do NOT require identical attention or answers.\n")

    print("32 query heads, head_dim=128, one layer, one token, FP16:")
    print(f"{'scheme':10}{'KV heads':>10}{'Q per KV':>12}{'cache bytes':>14}")
    for name, kv_heads in (("MHA", 32), ("GQA", 8), ("MQA", 1)):
        print(f"{name:10}{kv_heads:10}{32 // kv_heads:12}"
              f"{cache_bytes(1, kv_heads, 128, 1):14}")
    print("\nExample mapping for GQA with 8 query heads and 2 KV heads:")
    for head in range(8):
        print(f"  query head {head} -> KV head {head // 4}")
    print("\nGQA changes the architecture and learned projection shapes.")
    print("It is not a safe inference-time toggle for an arbitrary MHA model.")
    print("Quality comparisons need trained models, not random correlations.")
    print("Memory savings compare equal layer counts, head sizes and precision.")
    print("Llama-2-7B vs Qwen3-8B differs in layer count: 4/1.125 = 3.56x,")
    print("not exactly 4x. The isolated change from 32 to 8 KV heads is 4x.")


if __name__ == "__main__":
    main()
