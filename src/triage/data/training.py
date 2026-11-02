"""Turn the teacher's labels into the training files for the Kaggle notebook (Step 07).

Each training example is one conversation:
    prompt     = [a system message (our SYSTEM_PROMPT), a user message (the ticket)]
    completion = [an assistant message (the teacher's JSON answer)]
The model is trained to write the completion when it is given the prompt. Only the completion
counts when measuring the error (the prompt is just context).

IMPORTANT: the prompt is built with build_messages() from prompts.py. That is the SAME text the
evaluation code sends and the server will send. If we train on one prompt and serve with another,
quality silently drops.

Output folder (upload it to Kaggle as a Dataset):
    train.jsonl         train + train_extra + train_extra2 (with the merged labels)
    val.jsonl           only used to watch the error while training
    gold_prompts.jsonl  the gold tickets as ready prompts with NO labels, so the notebook can answer
                        them and we can score the result on this laptop
"""

import json
import logging

from triage import paths
from triage.io import read_jsonl, write_jsonl
from triage.prompts import build_messages
from triage.schema import TicketTriage
from triage.textclean import normalise

log = logging.getLogger(__name__)

TRAIN_PARTS = ["train", "train_extra", "train_extra2"]


def make_examples(label_file, texts):
    """Turn one merged label file into a list of prompt/completion examples.

    `texts` maps a ticket id to the ticket text.
    """
    examples = []
    for row in read_jsonl(label_file):
        # Check the label again and write it out again, so the key order is always the schema's.
        label = TicketTriage.model_validate(row["label"])
        answer = json.dumps(label.model_dump(mode="json"), ensure_ascii=False)

        example = {}
        example["id"] = row["id"]
        example["prompt"] = build_messages(texts[row["id"]])
        example["completion"] = [{"role": "assistant", "content": answer}]
        example["category"] = label.category.value  # only used for the statistics below
        examples.append(example)
    return examples


def check_no_leakage(train, val, test_rows, texts):
    """The model must never see a test ticket, a validation ticket, or a copy of one."""
    # Collect what is in the test set.
    test_ids = set()
    test_texts = set()
    for row in test_rows:
        test_ids.add(row["id"])
        test_texts.add(normalise(row["text"]))

    # Collect what is in the validation set.
    val_texts = set()
    for example in val:
        val_texts.add(normalise(texts[example["id"]]))

    # Check every training example against both.
    seen_ids = set()
    for example in train:
        text = normalise(texts[example["id"]])
        assert example["id"] not in test_ids, "a training ticket is in the test set"
        assert text not in test_texts, "a training text copies a test text"
        assert text not in val_texts, "a training text copies a validation text"
        assert example["id"] not in seen_ids, "duplicate training id"
        seen_ids.add(example["id"])


def build_training_files(merged_dir=paths.MERGED_LABELS, out_dir=paths.TRAINING):
    """Write train.jsonl, val.jsonl and gold_prompts.jsonl into `out_dir`."""
    # The ticket texts live in the processed files, the labels in the merged label files.
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
    log.info("Safety checks passed: no test or validation ticket (or copy of one) in the training set.")

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    # The gold tickets as ready prompts, with no labels.
    gold_prompts = []
    for row in test_rows:
        if row["gold"]:
            gold_prompts.append({"id": row["id"], "messages": build_messages(row["text"])})
    write_jsonl(out_dir / "gold_prompts.jsonl", gold_prompts)

    # Numbers worth looking at before spending any GPU time.
    log.info("train: %d examples   val: %d   gold prompts: %d", len(train), len(val), len(gold_prompts))

    counts = {}  # category -> how many training examples
    for example in train:
        category = example["category"]
        counts[category] = counts.get(category, 0) + 1
    log.info("train categories: %s", counts)

    lengths = []
    for example in train:
        characters = len(example["prompt"][0]["content"])
        characters = characters + len(example["prompt"][1]["content"])
        characters = characters + len(example["completion"][0]["content"])
        lengths.append(characters)
    lengths.sort()
    shortest = lengths[0]
    middle = lengths[len(lengths) // 2]
    longest = lengths[-1]
    log.info(
        "example size in characters: min %d, median %d, max %d (about %d tokens at the max)",
        shortest,
        middle,
        longest,
        longest // 4,
    )
    log.info("Wrote -> %s", out_dir)
