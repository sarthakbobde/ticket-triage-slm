"""Clean, remove near-duplicates and split the tickets BEFORE labelling (ADR-012).

Why before labelling: the splits decide which ~3,300 tickets we pay the teacher to label.

Outputs (unlabelled; the teacher adds the labels in triage.labeling.teacher)
- data/processed/test.jsonl     500 tickets, natural mix of categories, 200 flagged `gold` (FROZEN)
- data/processed/val.jsonl      300 tickets, natural mix of categories
- data/processed/train.jsonl    about 2,500 tickets, at most `cap` per category
- docs/reports/splits.md        counts per split and category (commit this)

WARNING: dataset_map.py now maps Human Resources to customer_service (ADR-013), so running this
again would draw a different sample than the one we labelled. The test set is frozen: the
function refuses to overwrite it unless force=True.
"""

import logging

import pandas as pd

from triage import paths
from triage.dataset_map import QUEUE_TO_CATEGORY
from triage.errors import UserError
from triage.textclean import clean_text, normalise

log = logging.getLogger(__name__)

SEED = 42
N_TEST = 500
N_GOLD = 200  # a part of the test set, labelled by hand
N_VAL = 300
TRAIN_TARGET = 2500  # the cap per category is worked out from this (see capped_sample)


def stratified_sample(df, n, by):
    """Pick exactly n rows, keeping every group's share of the whole.

    "Stratified" means each category keeps its natural share. We round every group's quota down,
    then give the missing rows to the groups with the biggest leftover (largest-remainder rounding),
    so the total is exactly n.
    """
    exact = df[by].value_counts(normalize=True) * n  # the exact (fractional) quota per group
    quota = exact.astype(int)  # rounded down

    # How many rows are still missing after rounding down, and which groups had the biggest leftover?
    leftovers = exact - quota
    biggest_first = leftovers.sort_values(ascending=False)
    missing = n - quota.sum()
    groups_that_get_one_more = biggest_first.index[:missing]  # [:missing] = the first `missing` groups
    for group in groups_that_get_one_more:
        quota[group] = quota[group] + 1

    # Take a random sample of that size from every group.
    parts = []
    for group, how_many in quota.items():
        if how_many > 0:
            rows_of_group = df[df[by] == group]
            parts.append(rows_of_group.sample(how_many, random_state=SEED))
    return pd.concat(parts)


def capped_sample(df, total, by):
    """Take at most `cap` rows per group, where the cap is chosen so the total is about `total`.

    Small groups are taken whole. What is left of the budget is shared by the larger groups
    ("water-filling"). So rare categories end up over-represented compared with reality, but not wildly.
    Returns (the sampled rows, the cap that was used).
    """
    sizes = df[by].value_counts().sort_values()  # smallest group first
    budget = total
    groups_left = len(sizes)
    cap = budget // groups_left  # // is whole-number division
    for size in sizes:
        cap = budget // groups_left
        if size >= cap:
            break  # this group (and every bigger one) is capped
        budget = budget - size  # this small group is taken whole, so it uses `size` of the budget
        groups_left = groups_left - 1

    parts = []
    for group_name, rows_of_group in df.groupby(by):
        how_many = min(len(rows_of_group), cap)
        parts.append(rows_of_group.sample(how_many, random_state=SEED))
    return pd.concat(parts), cap


def counts_table(splits):
    """A markdown table: how many tickets each category has in each split."""
    columns = {}
    for name in splits:
        columns[name] = splits[name]["category"].value_counts()
    counts = pd.DataFrame(columns).fillna(0)
    counts = counts.astype(int).sort_values("test", ascending=False)

    lines = []
    lines.append("| category | " + " | ".join(counts.columns) + " |")
    lines.append("|---" * (len(counts.columns) + 1) + "|")
    for category, row in counts.iterrows():
        cells = []
        for value in row:
            cells.append(str(value))
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    totals = []
    for value in counts.sum():
        totals.append(str(value))
    lines.append("| **total** | " + " | ".join(totals) + " |")
    return "\n".join(lines)


def build_splits(raw_file=paths.RAW_TICKETS, out_dir=paths.PROCESSED, report_file=None, force=False):
    """Build and save the test, validation and training splits."""
    if report_file is None:
        report_file = paths.DOC_REPORTS / "splits.md"

    test_path = out_dir / "test.jsonl"
    if test_path.exists() and not force:
        raise UserError(
            f"{test_path} exists and is frozen. Use --force only if you have not labelled "
            "or evaluated anything with it yet."
        )

    df = pd.read_json(raw_file, lines=True)
    df["text"] = df["text"].map(clean_text)

    # The dataset's queue, translated to our category name. Only used to keep the mix balanced.
    queue_to_category = {}
    for queue in QUEUE_TO_CATEGORY:
        queue_to_category[queue] = QUEUE_TO_CATEGORY[queue].value
    df["category"] = df["queue"].map(queue_to_category)
    log.info("Loaded %d tickets", len(df))

    # Step 1: remove near-duplicates BEFORE splitting, so a copy cannot end up in train while its
    # twin is in test.
    n_before = len(df)
    cleaned_texts = df["text"].map(normalise)
    is_repeat = cleaned_texts.duplicated()  # True for every copy after the first one
    df = df[~is_repeat]  # ~ means "not": keep only the rows that are not repeats
    n_near = n_before - len(df)
    log.info("Removed %d near-duplicates -> %d", n_near, len(df))

    # Step 2: the test set first, with the natural mix. Step 3: validation, natural mix.
    # Step 4: training from what is left, with at most `cap` per category.
    test = stratified_sample(df, N_TEST, "category")
    rest = df.drop(test.index)
    val = stratified_sample(rest, N_VAL, "category")
    rest = rest.drop(val.index)
    train, cap = capped_sample(rest, TRAIN_TARGET, "category")

    # Step 5: the gold set is a part of the test set, labelled by hand before seeing the teacher's labels.
    gold_sample = stratified_sample(test, N_GOLD, "category")
    gold_ids = set(gold_sample["id"])
    test["gold"] = test["id"].isin(gold_ids)

    # Save the three splits.
    keep = ["id", "text", "category", "priority", "queue", "version"]
    out_dir.mkdir(parents=True, exist_ok=True)
    split_frames = {"test": test, "val": val, "train": train}
    for name in ["test", "val", "train"]:
        split = split_frames[name]
        columns = list(keep)
        if name == "test":
            columns.append("gold")
        renamed = split[columns].rename(columns={"category": "dataset_category", "priority": "dataset_priority"})
        shuffled = renamed.sample(frac=1, random_state=SEED)  # frac=1 means "all rows, in random order"
        shuffled.to_json(out_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        log.info("Wrote %s: %d", name, len(split))

    # Write the report.
    gold_rows = test[test["gold"]]
    splits = {"test": test, "gold": gold_rows, "val": val, "train": train}
    report = []
    report.append("# Splits")
    report.append("")
    report.append("_Generated by `triage data splits` (ADR-012). Do not edit by hand._")
    report.append("")
    report.append(f"- Tickets after near-duplicate removal: **{len(df):,}** ({n_near} removed)")
    report.append(f"- Train cap per category: **{cap}** (derived from a target of {TRAIN_TARGET:,})")
    report.append(f"- Tickets to label with the teacher: **{len(test) + len(val) + len(train):,}**")
    report.append("- Categories are the dataset's queues mapped by `src/triage/dataset_map.py`: used only")
    report.append("  to stratify; the teacher's labels are the real targets.")
    report.append("")
    report.append(counts_table(splits))
    report.append("")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(report), encoding="utf-8")
    log.info("Wrote %s", report_file)
