import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.data import build, dataset_digest, load_split, validate_splits
from pipeline.evaluate import main as evaluate_main, parse_category, percentile, read_predictions, score


class DataAndEvaluationTests(unittest.TestCase):
    def test_dataset_is_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a", Path(folder) / "b"
            self.assertEqual(build(a), build(b))
            original_digest = dataset_digest(a)
            for path in b.glob("*.jsonl"):
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(original_digest, dataset_digest(b))
            splits = {name: load_split(a, name) for name in ("train", "validation", "test")}
            validate_splits(splits)
            self.assertEqual([len(rows) for rows in splits.values()], [72, 36, 36])
            splits["test"][0]["template_id"] = splits["train"][0]["template_id"]
            with self.assertRaisesRegex(ValueError, "Template leaks"):
                validate_splits(splits)
            with self.assertRaises(FileExistsError):
                build(a)

    def test_strict_json_scoring(self):
        self.assertEqual(parse_category('{"category":"billing"}'), "billing")
        for value in ('billing', '```json\n{"category":"billing"}\n```',
                      '{"category":"billing","extra":1}', '{"category":[]}', '[]', 'null'):
            self.assertIsNone(parse_category(value))

    def test_missing_and_invalid_are_not_dropped(self):
        examples = [{"id": "a", "category": "billing"}, {"id": "b", "category": "account"},
                    {"id": "c", "category": "technical"}]
        result = score(examples, {"a": {"prediction": '{"category":"billing"}', "latency_s": 1},
                                  "b": {"prediction": "account", "finish_reason": "length"}})
        self.assertEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["length_limited"], 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            score(examples, {"unknown": {"prediction": ""}})

    def test_percentiles_and_duplicate_predictions(self):
        self.assertEqual(percentile([1, 3], 95), 2.9)
        self.assertIsNone(percentile([], 95))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            row = json.dumps({"id": "same", "prediction": ""}) + "\n"
            path.write_text(row + row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_predictions(path)

    def test_prediction_manifest_must_match_dataset_and_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build(root / "data")
            example = load_split(root / "data", "test")[0]
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"id": example["id"], "prediction": example["response"]}) + "\n",
                                   encoding="utf-8")
            metadata = predictions.with_suffix(".jsonl.manifest.json")
            manifest = {"dataset_sha256": dataset_digest(root / "data"), "split": "train"}
            metadata.write_text(json.dumps(manifest), encoding="utf-8")
            args = ["evaluate", "--data", str(root / "data"), "--predictions", str(predictions),
                    "--output", str(root / "score.json")]
            with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "different dataset or split"):
                evaluate_main()
            self.assertFalse((root / "score.json").exists())
            manifest["split"] = "test"
            metadata.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("sys.argv", args), redirect_stdout(io.StringIO()):
                evaluate_main()
            result = json.loads((root / "score.json").read_text(encoding="utf-8"))
            self.assertEqual(result["correct"], 1)
            self.assertEqual(result["prediction_manifest"], manifest)


if __name__ == "__main__":
    unittest.main()
