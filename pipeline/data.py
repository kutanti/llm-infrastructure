"""Create a small, deliberately synthetic ticket-routing dataset."""

import argparse
import hashlib
import json
from pathlib import Path


CATEGORIES = ("billing", "account", "technical")
INSTRUCTION = (
    'Route this support ticket. Reply only with JSON {"category":"billing"}, '
    '{"category":"account"}, or {"category":"technical"}.\nTicket: '
)
TEMPLATES = {
    "train": {
        "billing": ("I was charged twice for {item}.", "Please refund my payment for {item}.",
                    "The invoice for {item} has the wrong amount.", "Explain the extra fee for {item}."),
        "account": ("I forgot my password for {item}.", "Change the email on my {item} account.",
                    "Unlock my {item} account.", "Help me reset my login for {item}."),
        "technical": ("The {item} app crashes on startup.", "The {item} page displays an error.",
                      "My {item} download keeps failing.", "The {item} screen freezes when I open it."),
    },
    "validation": {
        "billing": ("Why are there two charges for {item}?", "Can you return the money paid for {item}?"),
        "account": ("My {item} password needs a reset.", "I need a new login email for {item}."),
        "technical": ("Opening {item} causes a crash.", "An error prevents the {item} download."),
    },
    "test": {
        "billing": ("There is a duplicate payment for {item}.", "The fee on my {item} invoice is incorrect."),
        "account": ("I cannot remember the {item} password.", "My {item} login has been locked."),
        "technical": ("The {item} application stops responding.", "I see an error whenever {item} starts."),
    },
}
ITEMS = ("notebook", "calendar", "planner", "editor", "dashboard", "workspace")


def response_for(category):
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    return json.dumps({"category": category}, separators=(",", ":"))


def load_split(data_dir, split):
    if split not in TEMPLATES:
        raise ValueError("Split must be train, validation, or test")
    path = Path(data_dir) / f"{split}.jsonl"
    records = []
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            for field in ("id", "prompt", "response", "category", "split", "template_id"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"{path}:{line_number}: missing/non-string {field}")
            if row["split"] != split or row["id"] in ids:
                raise ValueError(f"{path}:{line_number}: wrong split or duplicate ID")
            if row["response"] != response_for(row["category"]):
                raise ValueError(f"{path}:{line_number}: response disagrees with category")
            ids.add(row["id"])
            records.append(row)
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records


def dataset_digest(data_dir):
    digest = hashlib.sha256()
    for split in TEMPLATES:
        path = Path(data_dir) / f"{split}.jsonl"
        digest.update(split.encode("ascii"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def validate_splits(splits):
    seen_ids, seen_prompts, seen_templates = set(), set(), set()
    for split, records in splits.items():
        templates = set()
        for row in records:
            normalized = " ".join(row["prompt"].lower().split())
            if row["id"] in seen_ids or normalized in seen_prompts:
                raise ValueError(f"Duplicate record across splits: {row['id']}")
            if row["template_id"] in seen_templates:
                raise ValueError(f"Template leaks across splits: {row['template_id']}")
            if row["split"] != split:
                raise ValueError(f"Wrong split for {row['id']}")
            seen_ids.add(row["id"])
            seen_prompts.add(normalized)
            templates.add(row["template_id"])
        seen_templates.update(templates)


def build(output):
    splits = {}
    for split, categories in TEMPLATES.items():
        records = []
        for category, templates in categories.items():
            for template_number, template in enumerate(templates):
                template_id = f"{split}-{category}-{template_number}"
                for item_number, item in enumerate(ITEMS):
                    records.append({
                        "id": f"{template_id}-{item_number}",
                        "template_id": template_id,
                        "split": split,
                        "category": category,
                        "prompt": INSTRUCTION + template.format(item=item),
                        "response": response_for(category),
                    })
        splits[split] = records
    validate_splits(splits)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for split, records in splits.items():
        with (output / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    manifest = {
        "schema_version": 1,
        "task": "synthetic-ticket-routing",
        "counts": {split: len(rows) for split, rows in splits.items()},
        "sha256": dataset_digest(output),
        "split_policy": "Disjoint wording templates, shared product nouns and category vocabulary.",
        "limitations": "Small synthetic exercise; not evidence of real support-ticket generalization.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results") / "pipeline" / "data")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
