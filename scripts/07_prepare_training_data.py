"""Step 07a: turn the teacher's labels into training files for the Kaggle notebook.

Each training example is one conversation:
    prompt     = [system message (our SYSTEM_PROMPT), user message (the ticket)]
    completion = [assistant message (the teacher's JSON answer)]
The model is trained to produce the completion given the prompt, and only the completion
counts for the loss (the prompt is just context).

IMPORTANT: the prompt is built with build_messages() from src/triage/prompts.py, the SAME text
the teacher saw (apart from the v1.1 category list), the eval harness sends, and the server will
send. Train on one prompt and serve with another and quality silently drops.

Writes to data/training/ (upload this folder to Kaggle as a Dataset):
    train.jsonl         1,540 examples: train + train_extra + train_extra2 (merged labels)
    val.jsonl           300 examples, only used to watch the loss during training
    gold_prompts.jsonl  the 200 gold tickets as ready-made prompts, NO labels (so the notebook
                        can answer them and we can score the result locally)

Run:  uv run python scripts/07_prepare_training_data.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

from triage.prompts import build_messages
from triage.schema import TicketTriage
from triage.textclean import normalise

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MERGED = ROOT / "data" / "labeled" / "merged"
OUT_DIR = ROOT / "data" / "training"

TRAIN_PARTS = ["train", "train_extra", "train_extra2"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_examples(label_file, texts):
    """One merged label file -> a list of prompt/completion examples."""
    examples = []
    for row in read_jsonl(label_file):
        # Re-validate and re-dump so the key order and format are always the schema's.
        label = TicketTriage.model_validate(row["label"])
        answer = json.dumps(label.model_dump(mode="json"), ensure_ascii=False)
        examples.append({
            "id": row["id"],
            "prompt": build_messages(texts[row["id"]]),
            "completion": [{"role": "assistant", "content": answer}],
            "category": label.category.value,  # kept for the statistics below only
        })
    return examples


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ticket texts live in the processed files, labels in the merged label files.
    texts = {}
    for name in TRAIN_PARTS + ["val"]:
        for row in read_jsonl(PROCESSED / f"{name}.jsonl"):
            texts[row["id"]] = row["text"]

    train = []
    for name in TRAIN_PARTS:
        train = train + make_examples(MERGED / f"{name}.jsonl", texts)
    val = make_examples(MERGED / "val.jsonl", texts)

    # Safety checks: the model must never see a test ticket, or a copy of one.
    test_rows = read_jsonl(PROCESSED / "test.jsonl")
    test_ids = set()
    test_texts = set()
    for row in test_rows:
        test_ids.add(row["id"])
        test_texts.add(normalise(row["text"]))
    val_texts = set()
    for example in val:
        val_texts.add(normalise(texts[example["id"]]))
    train_ids = set()
    for example in train:
        assert example["id"] not in test_ids, "a training ticket is in the test set"
        assert normalise(texts[example["id"]]) not in test_texts, "a training text copies a test text"
        assert normalise(texts[example["id"]]) not in val_texts, "a training text copies a val text"
        assert example["id"] not in train_ids, "duplicate training id"
        train_ids.add(example["id"])
    print("Safety checks passed: no test or val ticket (or copy of one) in the training set.")

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)

    # Gold tickets as ready prompts, with no labels.
    gold_prompts = []
    for row in test_rows:
        if row["gold"]:
            gold_prompts.append({"id": row["id"], "messages": build_messages(row["text"])})
    write_jsonl(OUT_DIR / "gold_prompts.jsonl", gold_prompts)

    # Statistics worth looking at before spending GPU time.
    print(f"\ntrain: {len(train)} examples   val: {len(val)}   gold prompts: {len(gold_prompts)}")
    counts = Counter()
    for example in train:
        counts[example["category"]] += 1
    print("train categories:", dict(counts.most_common()))

    lengths = []
    for example in train:
        characters = len(example["prompt"][0]["content"]) + len(example["prompt"][1]["content"])
        characters = characters + len(example["completion"][0]["content"])
        lengths.append(characters)
    lengths.sort()
    print(f"example size in characters: min {lengths[0]}, median {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]}  (about {lengths[-1] // 4} tokens at the max)")
    for name in ["train.jsonl", "val.jsonl", "gold_prompts.jsonl"]:
        size_kb = (OUT_DIR / name).stat().st_size / 1024
        print(f"  {name}: {size_kb:.0f} KB")
    print(f"\nWrote -> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
