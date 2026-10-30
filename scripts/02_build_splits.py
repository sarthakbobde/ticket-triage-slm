"""Step 03a: clean, near-dedupe and split the tickets BEFORE labelling (ADR-012).

Run:  uv run python scripts/02_build_splits.py

Why before labelling: the splits decide which ~3,300 tickets we pay the teacher to label.

Outputs (unlabelled; the teacher fills in labels in scripts/03_label_with_teacher.py)
- data/processed/test.jsonl     500, natural distribution, 200 flagged `gold` (FROZEN)
- data/processed/val.jsonl      300, natural distribution
- data/processed/train.jsonl    ~2,500, capped per category
- docs/reports/splits.md        counts per split and category (commit this)

See docs/steps/03-teacher-labels.md.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from triage.dataset_map import QUEUE_TO_CATEGORY

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
N_TEST = 500
N_GOLD = 200  # subset of test, hand-labelled by you
N_VAL = 300
TRAIN_TARGET = 2500  # the per-category cap is derived from this (see capped_sample)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tickets_en.jsonl"
OUT_DIR = ROOT / "data" / "processed"
OUT_REPORT = ROOT / "docs" / "reports" / "splits.md"


def clean_text(text: str) -> str:
    """What the teacher AND the student will see, so it must be identical everywhere (ADR-011)."""
    text = text.replace("\\n", "\n").replace("\\t", " ")  # literal escape sequences in the data
    text = re.sub(r"<br\s*/?>", "\n", text)  # HTML line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise(text: str) -> str:
    """Key for near-duplicate detection: case, punctuation and spacing don't make a new ticket."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


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
    for size in sizes:
        cap = budget // groups_left
        if size >= cap:
            break
        budget -= size
        groups_left -= 1
    parts = [g.sample(min(len(g), cap), random_state=SEED) for _, g in df.groupby(by)]
    return pd.concat(parts), cap


def table(splits: dict[str, pd.DataFrame]) -> str:
    ct = pd.DataFrame({name: s["category"].value_counts() for name, s in splits.items()}).fillna(0)
    ct = ct.astype(int).sort_values("test", ascending=False)
    lines = ["| category | " + " | ".join(ct.columns) + " |", "|---" * (len(ct.columns) + 1) + "|"]
    lines += [f"| {c} | " + " | ".join(str(v) for v in row) + " |" for c, row in ct.iterrows()]
    lines.append("| **total** | " + " | ".join(str(v) for v in ct.sum()) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing test set (don't)")
    args = ap.parse_args()

    test_path = OUT_DIR / "test.jsonl"
    if test_path.exists() and not args.force:
        sys.exit(f"{test_path.relative_to(ROOT)} exists and is frozen. Use --force only if you "
                 "have not labelled or evaluated anything with it yet.")

    df = pd.read_json(RAW, lines=True)
    df["text"] = df["text"].map(clean_text)
    df["category"] = df["queue"].map({q: c.value for q, c in QUEUE_TO_CATEGORY.items()})
    print(f"Loaded {len(df):,} tickets")

    # 1. Near-duplicates go BEFORE any split, so copies can't straddle train and test.
    n_before = len(df)
    df = df[~df["text"].map(normalise).duplicated()]
    n_near = n_before - len(df)
    print(f"Removed {n_near:,} near-duplicates -> {len(df):,}")

    # 2. Test first, natural distribution. 3. Val, natural. 4. Train from the rest, capped.
    test = stratified_sample(df, N_TEST, "category")
    rest = df.drop(test.index)
    val = stratified_sample(rest, N_VAL, "category")
    rest = rest.drop(val.index)
    train, cap = capped_sample(rest, TRAIN_TARGET, "category")

    # 5. Gold set: a stratified subset of test that you label by hand before seeing the teacher's.
    gold_ids = set(stratified_sample(test, N_GOLD, "category")["id"])
    test["gold"] = test["id"].isin(gold_ids)

    keep = ["id", "text", "category", "priority", "queue", "version"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in {"test": test, "val": val, "train": train}.items():
        cols = keep + (["gold"] if name == "test" else [])
        out = split[cols].rename(columns={"category": "dataset_category", "priority": "dataset_priority"})
        out.sample(frac=1, random_state=SEED).to_json(
            OUT_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False
        )
        print(f"Wrote {name}: {len(split):,}")

    splits = {"test": test, "gold": test[test["gold"]], "val": val, "train": train}
    report = [
        "# Splits",
        "",
        "_Generated by `scripts/02_build_splits.py` (ADR-012). Do not edit by hand._",
        "",
        f"- Tickets after near-duplicate removal: **{len(df):,}** ({n_near} removed)",
        f"- Train cap per category: **{cap}** (derived from a target of {TRAIN_TARGET:,})",
        f"- Tickets to label with the teacher: **{len(test) + len(val) + len(train):,}**",
        "- Categories are the dataset's queues mapped by `src/triage/dataset_map.py`: used only",
        "  to stratify; the teacher's labels are the real targets.",
        "",
        table(splits),
        "",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
