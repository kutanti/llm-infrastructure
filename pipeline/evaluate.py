"""Score exactly one prediction per held-out ID, retaining invalid outputs."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from pipeline.data import CATEGORIES, dataset_digest, load_split


def percentile(values, percent):
    if not values:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    ordered = sorted(values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("Percentiles require finite measurements")
    position = (len(ordered) - 1) * percent / 100
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def parse_category(prediction):
    if not isinstance(prediction, str):
        return None
    try:
        value = json.loads(prediction)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"category"}:
        return None
    return value["category"] if value["category"] in CATEGORIES else None


def read_predictions(path):
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{line_number}: missing string ID")
            if row["id"] in records:
                raise ValueError(f"{path}:{line_number}: duplicate prediction ID {row['id']}")
            if not isinstance(row.get("prediction"), str):
                raise ValueError(f"{path}:{line_number}: missing string prediction")
            records[row["id"]] = row
    return records


def score(examples, predictions):
    expected = {row["id"] for row in examples}
    unknown = set(predictions) - expected
    if unknown:
        raise ValueError(f"Prediction IDs outside selected split: {sorted(unknown)[:5]}")
    confusion = {category: {label: 0 for label in (*CATEGORIES, "invalid", "missing")}
                 for category in CATEGORIES}
    correct = valid = missing = truncated = context_limited = 0
    latencies = []
    for example in examples:
        row = predictions.get(example["id"])
        if row is None:
            missing += 1
            label = "missing"
        else:
            category = parse_category(row["prediction"])
            valid += category is not None
            label = category if category is not None else "invalid"
            truncated += row.get("finish_reason") in ("length", "context_length")
            context_limited += row.get("finish_reason") == "context_length"
            if "latency_s" in row:
                latency = row["latency_s"]
                if isinstance(latency, bool) or not isinstance(latency, (float, int)):
                    raise ValueError(f"Non-numeric latency for {example['id']}")
                if not math.isfinite(latency) or latency < 0:
                    raise ValueError(f"Invalid latency for {example['id']}")
                latencies.append(latency)
        confusion[example["category"]][label] += 1
        correct += label == example["category"]
    if not examples:
        raise ValueError("Cannot score an empty split")
    return {
        "examples": len(examples),
        "predictions": len(predictions),
        "correct": correct,
        "accuracy": correct / len(examples),
        "valid_json_rate": valid / len(examples),
        "missing": missing,
        "invalid": len(predictions) - valid,
        "length_limited": truncated,
        "context_limited": context_limited,
        "confusion": confusion,
        "latency_samples": len(latencies),
        "latency_s": {"mean": mean(latencies) if latencies else None,
                      "p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results") / "pipeline" / "data")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(load_split(args.data, args.split), read_predictions(args.predictions))
    result.update({"split": args.split, "dataset_sha256": dataset_digest(args.data),
                   "prediction_file": args.predictions.name})
    metadata_path = args.predictions.with_suffix(args.predictions.suffix + ".manifest.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_sha256") != result["dataset_sha256"] or metadata.get("split") != args.split:
            raise ValueError("Prediction manifest belongs to a different dataset or split")
        result["prediction_manifest"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
