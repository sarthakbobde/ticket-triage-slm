"""Clean, near-dedupe and split the tickets BEFORE labelling (ADR-012).

Why before labelling: the splits decide which ~3,300 tickets we pay the teacher to label.

Outputs (unlabelled; the teacher fills in labels in triage.labeling.teacher)
- data/processed/test.jsonl     500, natural distribution, 200 flagged `gold` (FROZEN)
- data/processed/val.jsonl      300, natural distribution
- data/processed/train.jsonl    ~2,500, capped per category
- docs/reports/splits.md        counts per split and category (commit this)

WARNING: dataset_map.py now maps Human Resources to customer_service (ADR-013), so running this
again would draw a different sample than the one we labelled. The test set is frozen: the
function refuses to overwrite it unless force=True.
"""

import logging
from pathlib import Path

import pandas as pd

from triage import paths
from triage.dataset_map import QUEUE_TO_CATEGORY
from triage.errors import UserError
from triage.textclean import clean_text, normalise

log = logging.getLogger(__name__)

SEED = 42
N_TEST = 500
N_GOLD = 200  # subset of test, hand-labelled
N_VAL = 300
TRAIN_TARGET = 2500  # the per-category cap is derived from this (see capped_sample)


def stratified_sample(df: pd.DataFrame, n: int, by: str) -> pd.DataFrame:
    """n rows keeping each group's share. Largest-remainder rounding so the total is exactly n."""
    exact = df[by].value_counts(normalize=True) * n
    quota = exact.astype(int)
    quota[(exact - quota).sort_values(ascending=False).index[: n - quota.sum()]] += 1
    parts = [df[df[by] == g].sample(q, random_state=SEED) for g, q in quota.items() if q]
    return pd.concat(parts)


def capped_sample(df: pd.DataFrame, total: int, by: str) -> tuple[pd.DataFrame, int]:
    """Take up to `cap` rows per group, with the cap chosen so the total is ~`total`.

    Small groups are taken whole; the leftover budget is shared by the large ones
    ("water-filling"). Rare classes end up over-represented relative to reality, not wildly.
    """
    sizes = df[by].value_counts().sort_values()
    budget, groups_left = total, len(sizes)
    cap = budget // groups_left
    for size in sizes:
        cap = budget // groups_left
        if size >= cap:
            break
        budget -= size
        groups_left -= 1
    parts = [g.sample(min(len(g), cap), random_state=SEED) for _, g in df.groupby(by)]
    return pd.concat(parts), cap


def counts_table(splits: dict[str, pd.DataFrame]) -> str:
    """Markdown table: tickets per category in each split."""
    ct = pd.DataFrame({name: s["category"].value_counts() for name, s in splits.items()}).fillna(0)
    ct = ct.astype(int).sort_values("test", ascending=False)
    lines = ["| category | " + " | ".join(ct.columns) + " |", "|---" * (len(ct.columns) + 1) + "|"]
    lines += [f"| {c} | " + " | ".join(str(v) for v in row) + " |" for c, row in ct.iterrows()]
    lines.append("| **total** | " + " | ".join(str(v) for v in ct.sum()) + " |")
    return "\n".join(lines)


def build_splits(
    raw_file: Path = paths.RAW_TICKETS,
    out_dir: Path = paths.PROCESSED,
    report_file: Path = paths.DOC_REPORTS / "splits.md",
    force: bool = False,
) -> None:
    test_path = out_dir / "test.jsonl"
    if test_path.exists() and not force:
        raise UserError(
            f"{test_path} exists and is frozen. Use --force only if you have not labelled "
            "or evaluated anything with it yet."
        )

    df = pd.read_json(raw_file, lines=True)
    df["text"] = df["text"].map(clean_text)
    df["category"] = df["queue"].map({q: c.value for q, c in QUEUE_TO_CATEGORY.items()})
    log.info("Loaded %d tickets", len(df))

    # 1. Near-duplicates go BEFORE any split, so copies can't straddle train and test.
    n_before = len(df)
    df = df[~df["text"].map(normalise).duplicated()]
    n_near = n_before - len(df)
    log.info("Removed %d near-duplicates -> %d", n_near, len(df))

    # 2. Test first, natural distribution. 3. Val, natural. 4. Train from the rest, capped.
    test = stratified_sample(df, N_TEST, "category")
    rest = df.drop(test.index)
    val = stratified_sample(rest, N_VAL, "category")
    rest = rest.drop(val.index)
    train, cap = capped_sample(rest, TRAIN_TARGET, "category")

    # 5. Gold set: a stratified subset of test, labelled by hand before seeing the teacher's.
    gold_ids = set(stratified_sample(test, N_GOLD, "category")["id"])
    test["gold"] = test["id"].isin(gold_ids)

    keep = ["id", "text", "category", "priority", "queue", "version"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in {"test": test, "val": val, "train": train}.items():
        cols = keep + (["gold"] if name == "test" else [])
        out = split[cols].rename(columns={"category": "dataset_category", "priority": "dataset_priority"})
        out.sample(frac=1, random_state=SEED).to_json(
            out_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False
        )
        log.info("Wrote %s: %d", name, len(split))

    splits = {"test": test, "gold": test[test["gold"]], "val": val, "train": train}
    report = [
        "# Splits",
        "",
        "_Generated by `triage data splits` (ADR-012). Do not edit by hand._",
        "",
        f"- Tickets after near-duplicate removal: **{len(df):,}** ({n_near} removed)",
        f"- Train cap per category: **{cap}** (derived from a target of {TRAIN_TARGET:,})",
        f"- Tickets to label with the teacher: **{len(test) + len(val) + len(train):,}**",
        "- Categories are the dataset's queues mapped by `src/triage/dataset_map.py`: used only",
        "  to stratify; the teacher's labels are the real targets.",
        "",
        counts_table(splits),
        "",
    ]
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(report), encoding="utf-8")
    log.info("Wrote %s", report_file)
