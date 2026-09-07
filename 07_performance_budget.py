"""Lesson 7: a deliberately simplified memory-bandwidth roofline."""

import argparse
import math


def positive_float(text):
    number = float(text)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-gib", type=positive_float, default=3.0)
    parser.add_argument("--bandwidth-gbs", type=positive_float, default=100.0)
    parser.add_argument("--first-output", type=positive_float, default=0.5)
    parser.add_argument("--decode-tps", type=positive_float, default=12.0)
    parser.add_argument("--tokens", type=int, default=240)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    weight_bytes = args.weight_gib * 1024**3
    bytes_per_second = args.bandwidth_gbs * 1e9
    upper_bound = bytes_per_second / weight_bytes
    print("ILLUSTRATIVE inputs, not measured RTX 4050 bandwidth or model traffic.")
    print(f"Weight traffic assumed per token: {args.weight_gib:.3f} GiB")
    print(f"Bandwidth assumed: {args.bandwidth_gbs:.1f} decimal GB/s")
    print(f"Weight-only upper bound: {upper_bound:.2f} tokens/s")
    print("\nAssumptions: one sequence, one full weight read per token, no reuse,")
    print("no compute limit, no KV reads, no transfers, no kernel overhead.")
    print("Real decoding can be limited by any of those omitted costs.")
    print("MoE and batching invalidate the simple full-weight-per-token premise.")
    print("\nSequential answer-time estimate (first-output includes first token):")
    for count in (args.tokens, max(1, args.tokens // 2)):
        seconds = args.first_output + (count - 1) / args.decode_tps
        print(f"  {count} tokens at {args.decode_tps:g} tok/s -> about {seconds:.2f} s")
    print("Fewer useful output tokens may improve latency more than kernel tuning.")
    print("Tokens are not words. Shortening an answer can also reduce its utility.")


if __name__ == "__main__":
    main()
