"""Lesson 6: analyze recorded inference metrics without loading a model."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=Path(__file__).resolve().parent / "data" / "qwen-baseline.json",
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cases = report["cases"]
    print("Reconstructed seconds = token count / recorded tokens per second.")
    print("Durations were measured by the runtime; first output by the client.\n")
    print(f"{'case':18}{'first':>9}{'load':>9}{'prefill':>10}{'decode':>10}{'wall':>9}")
    for case in cases:
        prefill = case["prompt_tokens"] / case["prompt_tokens_per_s"]
        decode = case["generated_tokens"] / case["generation_tokens_per_s"]
        print(f"{case['name']:18}{case['first_output_s']:9.2f}"
              f"{case['load_s']:9.2f}{prefill:10.2f}{decode:10.2f}"
              f"{case['wall_s']:9.2f}")
    substantive = [c for c in cases if c["name"] in ("coding", "explanation")]
    total_tokens = sum(c["generated_tokens"] for c in substantive)
    decode_seconds = sum(
        c["generated_tokens"] / c["generation_tokens_per_s"]
        for c in substantive
    )
    print(f"\nCombined coding/explanation decode: {total_tokens / decode_seconds:.2f} tok/s")
    print("This is total tokens / total decode time, not a mean of the rates.")
    print("Two short responses cannot establish sustained throughput or p95.")
    print("\nFirst output is not the same as model loading.")
    print("The first request also spent about 41 seconds in reported prefill.")
    print("We did not isolate disk caching, lazy initialization or kernel setup.")
    print("Those are hypotheses to investigate, not measured causes.")
    print("\nGPU residency is not GPU utilization or efficiency.")
    print("100% GPU means placement, not that all hardware capacity is exploited.")
    print("Snapshots show memory after requests, not the high-water mark.")
    print("\nQuality caveats: the code supplied only positive assertions;")
    print("the explanation has 120 whitespace-separated words, not under 120.")
    print("Distillation does not require changing the student's architecture.")
    print("A quick fluent answer is not automatically a correct answer.")


if __name__ == "__main__":
    main()
