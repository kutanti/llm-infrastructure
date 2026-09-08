import json
from pathlib import Path
import tempfile
import unittest

from pipeline.data import build, load_split
from pipeline.distill import prepare


class DistillationTests(unittest.TestCase):
    def test_teacher_targets_do_not_replace_heldout_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build(root / "data")
            train = load_split(root / "data", "train")
            predictions = root / "teacher.jsonl"
            rows = [{"id": row["id"], "prediction": '{"category":"technical"}'} for row in train]
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest = prepare(root / "data", predictions, root / "distilled", "test-teacher-revision")
            self.assertAlmostEqual(manifest["teacher_agreement_with_gold"], 1 / 3)
            targets = load_split(root / "distilled", "train")
            self.assertTrue(all(row["category"] == "technical" for row in targets))
            self.assertEqual(targets[0]["gold_category"], "billing")
            self.assertEqual(load_split(root / "data", "test"), load_split(root / "distilled", "test"))

    def test_mismatched_teacher_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build(root / "data")
            row = load_split(root / "data", "train")[0]
            predictions = root / "teacher.jsonl"
            predictions.write_text(json.dumps({"id": row["id"], "prediction": row["response"]}) + "\n",
                                   encoding="utf-8")
            predictions.with_suffix(".jsonl.manifest.json").write_text(
                json.dumps({"dataset_sha256": "different", "split": "train"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Teacher prediction manifest"):
                prepare(root / "data", predictions, root / "distilled", "teacher-revision")
            self.assertFalse((root / "distilled").exists())


if __name__ == "__main__":
    unittest.main()
