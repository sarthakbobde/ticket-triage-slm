"""Every file and folder the project reads or writes, defined once.

Nothing else in the code builds a path from scratch, so moving a folder means editing one line.
ROOT is found from this file's own location, so it works from any working directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
MODELS_CONFIG = CONFIGS / "models.yaml"
ENV_FILE = ROOT / ".env"

DATA = ROOT / "data"
RAW_TICKETS = DATA / "raw" / "tickets_en.jsonl"
PROCESSED = DATA / "processed"  # test / val / train splits, no labels
LABELED = DATA / "labeled"  # raw teacher output
MERGED_LABELS = LABELED / "merged"  # teacher output after the hr merge (ADR-013)
GOLD = DATA / "gold"
GOLD_LABELS = GOLD / "gold_labels.jsonl"
TRAINING = DATA / "training"  # what gets uploaded to Kaggle

EVALS = ROOT / "evals"
EVAL_RUNS = EVALS / "runs"  # raw model answers (git-ignored)
EVAL_REPORTS = EVALS / "reports"  # scored reports (committed)

DOC_REPORTS = ROOT / "docs" / "reports"
