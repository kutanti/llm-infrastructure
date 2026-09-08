"""Predict the training split's most frequent category, without a language model."""

import argparse
from collections import Counter
import json
from pathlib import Path

from pipeline.data import CATEGORIES, load_split, response_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results") / "pipeline" / "data")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = Counter(row["category"] for row in load_split(args.data, "train"))
    category = max(CATEGORIES, key=lambda item: counts[item])
    rows = load_split(args.data, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps({"id": row["id"], "prediction": response_for(category)}) + "\n")
    print(f"Training-majority baseline: {category}; wrote {len(rows)} predictions.")


if __name__ == "__main__":
    main()
