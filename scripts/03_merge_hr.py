"""Step 03d: apply ADR-013 (hr merged into customer_service) to the teacher's labels.

The teacher labelled with the 6-category prompt (schema v1.0). We do NOT call it again.
We copy each labelled row and change one thing: category "hr" becomes "customer_service".
Every merged label is then checked against the v1.1 schema.

Input : data/labeled/<split>__gpt-4.1.jsonl   (never changed: it is the raw record)
Output: data/labeled/merged/<split>.jsonl     (what Step 04+ should use)

Free to run (no API calls):  uv run python scripts/03_merge_hr.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

from triage.schema import SCHEMA_VERSION, TicketTriage

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
LABELED = ROOT / "data" / "labeled"
MERGED = LABELED / "merged"
TEACHER = "gpt-4.1"
SPLITS = ["val", "train", "train_extra", "train_extra2"]


def main():
    MERGED.mkdir(parents=True, exist_ok=True)
    total_counts = Counter()  # categories over the three train files

    for split in SPLITS:
        in_path = LABELED / f"{split}__{TEACHER}.jsonl"
        out_path = MERGED / f"{split}.jsonl"
        if not in_path.exists():
            print(f"{split}: no file, skipped")
            continue

        counts = Counter()
        changed = 0
        with open(in_path, encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as out:
            for line in src:
                row = json.loads(line)
                label = row["label"]

                if label["category"] == "hr":
                    label["category"] = "customer_service"
                    row["merged_from"] = "hr"  # keep a trace of what the teacher said
                    changed = changed + 1

                # Raises an error if anything does not fit the v1.1 schema.
                TicketTriage.model_validate(label)

                row["schema_version"] = SCHEMA_VERSION
                counts[label["category"]] += 1
                if split != "val":
                    total_counts[label["category"]] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"{split}: {sum(counts.values())} rows, {changed} changed hr -> customer_service")
        print(f"   {dict(counts.most_common())}")

    print(f"\nTraining labels (train + extras): {sum(total_counts.values())}")
    for category, n in total_counts.most_common():
        print(f"   {category:<18}{n}")


if __name__ == "__main__":
    main()
