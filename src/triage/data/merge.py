"""Apply ADR-013 to the teacher's labels: the category hr is merged into customer_service.

The teacher labelled with the six-category prompt (schema v1.0). We do NOT call it again: we copy
each labelled row and change one thing. Every merged label is then checked against schema v1.1.

Input : data/labeled/<split>__<teacher>.jsonl   (never changed: the raw record)
Output: data/labeled/merged/<split>.jsonl       (what training and evaluation should use)
"""

import logging

from triage import paths
from triage.io import read_jsonl, write_jsonl
from triage.schema import SCHEMA_VERSION, TicketTriage

log = logging.getLogger(__name__)

SPLITS = ["val", "train", "train_extra", "train_extra2"]

# old category -> the category it becomes
MERGED_CATEGORIES = {"hr": "customer_service"}


def merge_hr_labels(teacher="gpt-4.1", labeled_dir=paths.LABELED, merged_dir=paths.MERGED_LABELS):
    """Write the merged label files. Returns, for each split, how many rows each category has."""
    counts_per_split = {}

    for split in SPLITS:
        in_path = labeled_dir / f"{split}__{teacher}.jsonl"
        if not in_path.exists():
            log.info("%s: no file, skipped", split)
            continue

        counts = {}  # category -> how many rows
        changed = 0
        merged_rows = []

        for row in read_jsonl(in_path):
            label = row["label"]

            # Change hr to customer_service, and keep a note of what the teacher originally said.
            if label["category"] in MERGED_CATEGORIES:
                row["merged_from"] = label["category"]
                label["category"] = MERGED_CATEGORIES[label["category"]]
                changed = changed + 1

            # This raises an error if the label does not fit schema v1.1.
            TicketTriage.model_validate(label)

            row["schema_version"] = SCHEMA_VERSION
            category = label["category"]
            counts[category] = counts.get(category, 0) + 1
            merged_rows.append(row)

        write_jsonl(merged_dir / f"{split}.jsonl", merged_rows)
        counts_per_split[split] = counts
        log.info("%s: %d rows, %d changed hr -> customer_service", split, len(merged_rows), changed)
        log.info("   %s", counts)

    return counts_per_split
