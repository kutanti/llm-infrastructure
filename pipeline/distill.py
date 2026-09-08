"""Build sequence-distillation targets from saved teacher responses."""

import argparse
import json
from pathlib import Path

from pipeline.data import dataset_digest, load_split, response_for
from pipeline.evaluate import parse_category, read_predictions


def prepare(data_dir, predictions_file, output, teacher_identity):
    if not teacher_identity.strip():
        raise ValueError("Record the teacher's model/revision or artifact digest")
    splits = {split: load_split(data_dir, split) for split in ("train", "validation", "test")}
    predictions = read_predictions(predictions_file)
    predictions_file = Path(predictions_file)
    metadata_path = predictions_file.with_suffix(predictions_file.suffix + ".manifest.json")
    teacher_metadata = None
    if metadata_path.exists():
        teacher_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (teacher_metadata.get("dataset_sha256") != dataset_digest(data_dir)
                or teacher_metadata.get("split") != "train"):
            raise ValueError("Teacher prediction manifest must match the source dataset's training split")
    expected = {row["id"] for row in splits["train"]}
    if set(predictions) - expected:
        raise ValueError("Teacher predictions must contain only training-split IDs")
    targets = []
    rejected = []
    agreements = 0
    for original in splits["train"]:
        prediction = predictions.get(original["id"])
        category = parse_category(prediction["prediction"]) if prediction is not None else None
        if category is None:
            rejected.append({"id": original["id"], "reason": "missing or invalid teacher response"})
            continue
        agreements += category == original["category"]
        targets.append({**original, "gold_category": original["category"],
                        "category": category, "response": response_for(category)})
    if not targets:
        raise ValueError("No valid teacher targets; do not train on an empty dataset")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits["train"] = targets
    for split, rows in splits.items():
        with (output / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    manifest = {
        "task": "synthetic-ticket-routing-sequence-distillation",
        "teacher": teacher_identity,
        "teacher_prediction_manifest": teacher_metadata,
        "source_dataset_sha256": dataset_digest(data_dir),
        "sha256": dataset_digest(output),
        "accepted": len(targets),
        "rejected": rejected,
        "teacher_agreement_with_gold": agreements / len(targets),
        "objective": "Next-token cross-entropy on teacher-written category JSON; not logit KL.",
        "policy": "Keep all valid teacher categories, including mistakes; validation/test retain gold labels.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results") / "pipeline" / "data")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--teacher", required=True, help="Teacher model/revision or artifact digest")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.predictions, args.output, args.teacher), indent=2))


if __name__ == "__main__":
    main()
