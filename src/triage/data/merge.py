"""Apply ADR-013 to the teacher's labels: hr is merged into customer_service.

The teacher labelled with the six-category prompt (schema v1.0). We do NOT call it again: we copy
each labelled row and change one thing. Every merged label is then checked against schema v1.1.

Input : data/labeled/<split>__<teacher>.jsonl   (never changed: the raw record)
Output: data/labeled/merged/<split>.jsonl       (what training and evaluation should use)
"""

import logging
from collections import Counter
from pathlib import Path

from triage import paths
from triage.io import read_jsonl, write_jsonl
from triage.schema import SCHEMA_VERSION, TicketTriage

log = logging.getLogger(__name__)

SPLITS = ["val", "train", "train_extra", "train_extra2"]
MERGED_CATEGORIES = {"hr": "customer_service"}


def merge_hr_labels(
    teacher: str = "gpt-4.1", labeled_dir: Path = paths.LABELED, merged_dir: Path = paths.MERGED_LABELS
) -> dict[str, Counter]:
    """Write merged label files. Returns the category counts per split."""
    counts_per_split = {}
    for split in SPLITS:
        in_path = labeled_dir / f"{split}__{teacher}.jsonl"
        if not in_path.exists():
            log.info("%s: no file, skipped", split)
            continue

        counts = Counter()
        changed = 0
        merged_rows = []
        for row in read_jsonl(in_path):
            label = row["label"]
            if label["category"] in MERGED_CATEGORIES:
                row["merged_from"] = label["category"]  # keep a trace of what the teacher said
                label["category"] = MERGED_CATEGORIES[label["category"]]
                changed = changed + 1

            TicketTriage.model_validate(label)  # raises if anything does not fit schema v1.1
            row["schema_version"] = SCHEMA_VERSION
            counts[label["category"]] += 1
            merged_rows.append(row)

        write_jsonl(merged_dir / f"{split}.jsonl", merged_rows)
        counts_per_split[split] = counts
        log.info("%s: %d rows, %d changed hr -> customer_service", split, len(merged_rows), changed)
        log.info("   %s", dict(counts.most_common()))
    return counts_per_split
