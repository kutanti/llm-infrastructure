"""Lesson 2: cache storage, block overhead, context, and parallel sequences."""

import argparse

from cache_math import FORMATS, GIB, MIB, SHAPES, cache_bytes


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=positive_int, default=8192)
    parser.add_argument("--sequences", type=positive_int, default=1)
    args = parser.parse_args()
    assert cache_bytes(36, 8, 128, 8192) == 1152 * MIB
    assert cache_bytes(36, 8, 128, 8192, key_type="q8_0") == 612 * MIB
    assert cache_bytes(8, 4, 256, 8192) == 256 * MIB

    print(f"Context={args.context}; independent sequences={args.sequences}")
    print("Conventional K/V tensor storage in MiB, NOT total model memory.\n")
    print(f"{'model':29}{'FP16':>12}{'Q8_0':>12}{'Q4_0':>12}")
    for shape in SHAPES:
        amounts = [
            cache_bytes(
                shape.full_attention_layers, shape.kv_heads, shape.head_dim,
                args.context, sequences=args.sequences, key_type=dtype,
            ) / MIB
            for dtype in FORMATS
        ]
        print(f"{shape.name:29}" + "".join(f"{n:12.2f}" for n in amounts))
        print(f"  Scope: {shape.scope}")

    print("\nBytes per element including block scales:")
    for name, (elements, size) in FORMATS.items():
        print(f"  {name}: {size}/{elements} = {size / elements:.4f}")
    print("\nQwen3.5-4B at 4K, 8K, 16K, 32K (FP16, one sequence):")
    for context in (4096, 8192, 16384, 32768):
        size = cache_bytes(8, 4, 256, context)
        print(f"  {context:6} tokens -> {size / MIB:7.1f} MiB")

    print(f"\nLaptop GPU capacity: 6141 MiB = {6141 * MIB / GIB:.4f} GiB")
    print("Do not infer FITS/OOM from this table alone. Also budget weights,")
    print("recurrent state, compute buffers, allocator overhead and other apps.")
    print("Shared-prefix caches can reduce duplication; padding can increase it.")
    print("These formulas do not establish quantized-cache support for hybrids.")
    print("Architecture sources and scope limitations: SOURCES.md and GUIDE.md.")


if __name__ == "__main__":
    main()
