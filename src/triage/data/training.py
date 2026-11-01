"""Turn the teacher's labels into the training files for the Kaggle notebook (Step 07).

Each training example is one conversation:
    prompt     = [system message (our SYSTEM_PROMPT), user message (the ticket)]
    completion = [assistant message (the teacher's JSON answer)]
The model is trained to produce the completion given the prompt, and only the completion
counts for the loss (the prompt is just context).

IMPORTANT: the prompt is built with build_messages() from prompts.py, the SAME text the eval
harness sends and the server will send. Train on one prompt and serve with another and quality
silently drops.

Output folder (upload it to Kaggle as a Dataset):
    train.jsonl         train + train_extra + train_extra2 (merged labels)
    val.jsonl           only used to watch the loss during training
    gold_prompts.jsonl  the gold tickets as ready-made prompts with NO labels, so the notebook
                        can answer them and we can score the result locally
"""

import json
import logging
from collections import Counter
from pathlib import Path

from triage import paths
from triage.io import Row, read_jsonl, write_jsonl
from triage.prompts import build_messages
from triage.schema import TicketTriage
from triage.textclean import normalise

log = logging.getLogger(__name__)

TRAIN_PARTS = ["train", "train_extra", "train_extra2"]


def make_examples(label_file: Path, texts: dict[str, str]) -> list[Row]:
    """One merged label file -> a list of prompt/completion examples."""
    examples = []
    for row in read_jsonl(label_file):
        # Re-validate and re-dump so the key order and format are always the schema's.
        label = TicketTriage.model_validate(row["label"])
        answer = json.dumps(label.model_dump(mode="json"), ensure_ascii=False)
        examples.append(
            {
                "id": row["id"],
                "prompt": build_messages(texts[row["id"]]),
                "completion": [{"role": "assistant", "content": answer}],
                "category": label.category.value,  # kept for statistics only
            }
        )
    return examples


def check_no_leakage(train: list[Row], val: list[Row], test_rows: list[Row], texts: dict[str, str]) -> None:
    """The model must never see a test ticket, a val ticket, or a copy of one."""
    test_ids = {row["id"] for row in test_rows}
    test_texts = {normalise(row["text"]) for row in test_rows}
    val_texts = {normalise(texts[example["id"]]) for example in val}
    seen_ids = set()
    for example in train:
        text = normalise(texts[example["id"]])
        assert example["id"] not in test_ids, "a training ticket is in the test set"
        assert text not in test_texts, "a training text copies a test text"
        assert text not in val_texts, "a training text copies a val text"
        assert example["id"] not in seen_ids, "duplicate training id"
        seen_ids.add(example["id"])


def build_training_files(merged_dir: Path = paths.MERGED_LABELS, out_dir: Path = paths.TRAINING) -> None:
    # Ticket texts live in the processed files, labels in the merged label files.
    texts = {}
    for name in TRAIN_PARTS + ["val"]:
        for row in read_jsonl(paths.PROCESSED / f"{name}.jsonl"):
            texts[row["id"]] = row["text"]

    train = []
    for name in TRAIN_PARTS:
        train = train + make_examples(merged_dir / f"{name}.jsonl", texts)
    val = make_examples(merged_dir / "val.jsonl", texts)

    test_rows = read_jsonl(paths.PROCESSED / "test.jsonl")
    check_no_leakage(train, val, test_rows, texts)
    log.info("Safety checks passed: no test or val ticket (or copy of one) in the training set.")

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    gold_prompts = []
    for row in test_rows:
        if row["gold"]:
            gold_prompts.append({"id": row["id"], "messages": build_messages(row["text"])})
    write_jsonl(out_dir / "gold_prompts.jsonl", gold_prompts)

    # Statistics worth looking at before spending GPU time.
    log.info("train: %d examples   val: %d   gold prompts: %d", len(train), len(val), len(gold_prompts))
    counts = Counter()
    for example in train:
        counts[example["category"]] += 1
    log.info("train categories: %s", dict(counts.most_common()))
    lengths = []
    for example in train:
        characters = len(example["prompt"][0]["content"]) + len(example["prompt"][1]["content"])
        characters = characters + len(example["completion"][0]["content"])
        lengths.append(characters)
    lengths.sort()
    log.info(
        "example size in characters: min %d, median %d, max %d (about %d tokens at the max)",
        lengths[0],
        lengths[len(lengths) // 2],
        lengths[-1],
        lengths[-1] // 4,
    )
    log.info("Wrote -> %s", out_dir)
