"""Prepare two fixed retrieval prompts with different amounts of input text."""

import argparse
from pathlib import Path


def make_prompt(record_count):
    if record_count < 1:
        raise ValueError("record_count must be positive")
    records = []
    for index in range(1, record_count + 1):
        shelf = (index * 7) % 97 + 1
        records.append(f"Record {index:03d}: item P{index:03d} is on shelf {shelf:02d}.")
    records.insert(
        record_count // 2,
        "Record TARGET: item KESTREL is on shelf 42.",
    )
    return (
        "Read this warehouse inventory and answer the question after it.\n\n"
        + "\n".join(records)
        + "\n\nWhich shelf holds item KESTREL? Reply with only the shelf number.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "retrieval")
    args = parser.parse_args()
    prompts = {
        "short.txt": make_prompt(16),
        "long.txt": make_prompt(192),
    }
    for name in prompts:
        if (args.out_dir / name).exists():
            parser.error(f"{args.out_dir / name} already exists; choose a new --out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in prompts.items():
        with (args.out_dir / name).open("x", encoding="utf-8") as output:
            output.write(text)
        print(f"{args.out_dir / name}: {len(text)} characters, {len(text.split())} words")
    print("Both answers should be 42. Runtime prompt_eval_count gives the token count.")
    print("The documents are fixed across runs, but no tokenizer is assumed here.")


if __name__ == "__main__":
    main()
