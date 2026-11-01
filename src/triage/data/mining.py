"""Find extra tickets that probably belong to the rare categories (hr, customer_service).

Why: the teacher found only ~1% hr and ~2% customer_service in the first 1,000 training tickets.
That is too few for the small model to learn from, so we search the unused tickets for more.

How: every unused ticket gets a score per category = how many of that category's keywords it
contains, and we keep the highest scorers. This only PICKS tickets to label. The teacher still
decides the real category, and many picked tickets turn out to be something else.

Two rounds exist. Round 1 is already done: running it again would skip everything it picked.
  Round 1: broad lists for hr and customer_service -> train_extra.jsonl
  Round 2: strict hr list built from the 34 confirmed hr tickets -> train_extra2.jsonl

Never touched: test.jsonl and val.jsonl (those stay natural and frozen).
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triage import paths
from triage.data.keywords import BROAD_KEYWORDS, STRICT_HR_KEYWORDS
from triage.dataset_map import QUEUE_TO_CATEGORY
from triage.errors import UserError
from triage.io import Row, read_jsonl, write_jsonl
from triage.textclean import clean_text, normalise

log = logging.getLogger(__name__)

SEED = 42


@dataclass
class MiningRound:
    keywords: dict[str, dict[str, list[str]]]
    how_many: dict[str, int]  # candidates to pick per category
    out_name: str


ROUNDS = {
    1: MiningRound(BROAD_KEYWORDS, {"hr": 200, "customer_service": 300}, "train_extra.jsonl"),
    2: MiningRound(STRICT_HR_KEYWORDS, {"hr": 150}, "train_extra2.jsonl"),
}


def split_into_words(text: str) -> list[str]:
    """'Hello, World!' becomes ['hello', 'world']. Letters and digits stay, the rest is a gap."""
    lower_text = text.lower()
    cleaned = ""
    for character in lower_text:
        if character.isalnum():
            cleaned = cleaned + character
        else:
            cleaned = cleaned + " "
    return cleaned.split()


def keyword_score(lower_text: str, ticket_words: list[str], keywords: dict[str, list[str]]) -> int:
    """How many different keywords from the list does this ticket contain?"""
    score = 0
    for word in keywords["words"]:
        if word in ticket_words:
            score = score + 1
    for phrase in keywords["phrases"]:
        if phrase in lower_text:
            score = score + 1
    return score


def get_score(match: dict[str, Any]) -> int:
    """Used for sorting: the sort looks at this number for each match."""
    return match["score"]


def used_ticket_ids() -> set[str]:
    """Test and val are off limits for ever; every training ticket we already paid to label is done."""
    files = [
        paths.PROCESSED / "test.jsonl",
        paths.PROCESSED / "val.jsonl",
        paths.LABELED / "train__gpt-4.1.jsonl",
        paths.LABELED / "train_extra__gpt-4.1.jsonl",
    ]
    ids = set()
    for path in files:
        if path.exists():
            for row in read_jsonl(path):
                ids.add(row["id"])
    return ids


def used_ticket_texts(used_ids: set[str]) -> set[str]:
    """Normalised texts of used tickets, so near-copies of them are skipped too."""
    texts = set()
    for name in ["test.jsonl", "val.jsonl", "train.jsonl", "train_extra.jsonl"]:
        path = paths.PROCESSED / name
        if path.exists():
            for row in read_jsonl(path):
                if row["id"] in used_ids:
                    texts.add(normalise(row["text"]))
    return texts


def mine_rare_classes(round_number: int, out_dir: Path = paths.PROCESSED) -> list[Row]:
    """Run one mining round, write its file, and return the picked tickets."""
    if round_number not in ROUNDS:
        raise UserError(f"Unknown round {round_number}. Choose from {sorted(ROUNDS)}.")
    config = ROUNDS[round_number]
    out_file = out_dir / config.out_name

    # 1. What is already used?
    used_ids = used_ticket_ids()
    used_texts = used_ticket_texts(used_ids)
    log.info("Already used: %d tickets", len(used_ids))

    # 2. Score every unused ticket for each category we are mining.
    candidates = []
    seen_texts = set()
    for ticket in read_jsonl(paths.RAW_TICKETS):
        if ticket["id"] in used_ids:
            continue
        text = clean_text(ticket["text"])
        key = normalise(text)
        if key in used_texts or key in seen_texts:
            continue
        seen_texts.add(key)

        lower_text = text.lower()
        ticket_words = split_into_words(text)
        scores = {}
        for category in config.keywords:
            scores[category] = keyword_score(lower_text, ticket_words, config.keywords[category])
        candidates.append(
            {
                "id": ticket["id"],
                "text": text,
                "dataset_category": QUEUE_TO_CATEGORY[ticket["queue"]].value,
                "dataset_priority": ticket["priority"],
                "queue": ticket["queue"],
                "version": ticket["version"],
                "scores": scores,
            }
        )
    log.info("Unused tickets to search: %d", len(candidates))

    # 3. For each category: keep tickets with at least one hit, best scores first.
    random.seed(SEED)
    picked_ids = set()
    picked = []
    for category in config.keywords:  # hr comes first: it is the rarest
        matches = []
        for ticket in candidates:
            if ticket["id"] in picked_ids:
                continue  # already picked for the other category
            score = ticket["scores"][category]
            if score >= 1:
                matches.append({"ticket": ticket, "score": score})

        random.shuffle(matches)  # so ties are broken randomly, not by file order
        matches.sort(key=get_score, reverse=True)  # stable sort: ties keep the random order

        log.info("%s: %d tickets contain at least one keyword", category, len(matches))
        for wanted_score in [5, 4, 3, 2, 1]:
            how_many = 0
            for match in matches:
                if match["score"] == wanted_score:
                    how_many = how_many + 1
            log.info("  with %d keyword hit(s): %d", wanted_score, how_many)

        limit = config.how_many[category]
        for match in matches[:limit]:
            ticket = match["ticket"]
            picked_ids.add(ticket["id"])
            picked.append(
                {
                    "id": ticket["id"],
                    "text": ticket["text"],
                    "dataset_category": ticket["dataset_category"],
                    "dataset_priority": ticket["dataset_priority"],
                    "queue": ticket["queue"],
                    "version": ticket["version"],
                    "mined_for": category,
                }
            )
        log.info("  picked %d", min(limit, len(matches)))

    # 4. Shuffle so the categories are mixed, then save.
    random.shuffle(picked)
    write_jsonl(out_file, picked)
    log.info("Wrote %d tickets -> %s", len(picked), out_file)
    return picked
