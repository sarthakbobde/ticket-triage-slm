"""Find extra tickets that probably belong to the rare categories (hr, customer_service).

Why: the teacher found only about 1% hr and 2% customer_service in the first 1,000 training tickets.
That is too few for the small model to learn from, so we search the unused tickets for more.

How: every unused ticket gets a score for each category: how many of that category's keywords it
contains. We keep the highest scorers. This only PICKS tickets to label. The teacher still decides
the real category, and many picked tickets turn out to be something else.

There are two rounds. Round 1 is already done: running it again would skip everything it picked.
  Round 1: broad lists for hr and customer_service -> train_extra.jsonl
  Round 2: a strict hr list built from the 34 confirmed hr tickets -> train_extra2.jsonl

Never touched: test.jsonl and val.jsonl (those stay natural and frozen).
"""

import logging
import random

from triage import paths
from triage.data.keywords import BROAD_KEYWORDS, STRICT_HR_KEYWORDS
from triage.dataset_map import QUEUE_TO_CATEGORY
from triage.errors import UserError
from triage.io import read_jsonl, write_jsonl
from triage.textclean import clean_text, normalise

log = logging.getLogger(__name__)

SEED = 42


class MiningRound:
    """The settings of one mining round."""

    def __init__(self, keywords, how_many, out_name):
        self.keywords = keywords  # category -> its keyword lists
        self.how_many = how_many  # category -> how many tickets to pick
        self.out_name = out_name  # the file the picked tickets are saved to


ROUNDS = {
    1: MiningRound(BROAD_KEYWORDS, {"hr": 200, "customer_service": 300}, "train_extra.jsonl"),
    2: MiningRound(STRICT_HR_KEYWORDS, {"hr": 150}, "train_extra2.jsonl"),
}


def split_into_words(text):
    """Cut a text into lower-case words. 'Hello, World!' becomes ['hello', 'world'].

    Letters and digits stay. Everything else is treated as a gap between words.
    """
    lower_text = text.lower()
    cleaned = ""
    for character in lower_text:
        if character.isalnum():
            cleaned = cleaned + character
        else:
            cleaned = cleaned + " "
    return cleaned.split()


def keyword_score(lower_text, ticket_words, keywords):
    """How many different keywords from the list does this ticket contain?

    A single word must match a WHOLE word ("hr" must not match "three").
    A phrase only has to appear somewhere in the text.
    """
    score = 0
    for word in keywords["words"]:
        if word in ticket_words:
            score = score + 1
    for phrase in keywords["phrases"]:
        if phrase in lower_text:
            score = score + 1
    return score


def get_score(match):
    """Used by the sort below: it sorts the matches by this number."""
    return match["score"]


def used_ticket_ids():
    """Ids of tickets we may not pick: test and validation (off limits for ever) and everything already labelled."""
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


def used_ticket_texts(used_ids):
    """The cleaned-up texts of used tickets, so near-copies of them are skipped too."""
    texts = set()
    for name in ["test.jsonl", "val.jsonl", "train.jsonl", "train_extra.jsonl"]:
        path = paths.PROCESSED / name
        if path.exists():
            for row in read_jsonl(path):
                if row["id"] in used_ids:
                    texts.add(normalise(row["text"]))
    return texts


def mine_rare_classes(round_number, out_dir=paths.PROCESSED):
    """Run one mining round, save the picked tickets, and return them."""
    if round_number not in ROUNDS:
        raise UserError(f"Unknown round {round_number}. Choose from {sorted(ROUNDS)}.")
    config = ROUNDS[round_number]
    out_file = out_dir / config.out_name

    # Step 1: which tickets are already used?
    used_ids = used_ticket_ids()
    used_texts = used_ticket_texts(used_ids)
    log.info("Already used: %d tickets", len(used_ids))

    # Step 2: give every unused ticket a score for each category we are mining.
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

        candidate = {}
        candidate["id"] = ticket["id"]
        candidate["text"] = text
        candidate["dataset_category"] = QUEUE_TO_CATEGORY[ticket["queue"]].value
        candidate["dataset_priority"] = ticket["priority"]
        candidate["queue"] = ticket["queue"]
        candidate["version"] = ticket["version"]
        candidate["scores"] = scores
        candidates.append(candidate)
    log.info("Unused tickets to search: %d", len(candidates))

    # Step 3: for each category, keep the tickets with at least one keyword, best scores first.
    random.seed(SEED)
    picked_ids = set()
    picked = []
    for category in config.keywords:  # hr comes first because it is the rarest
        matches = []
        for ticket in candidates:
            if ticket["id"] in picked_ids:
                continue  # already picked for the other category
            score = ticket["scores"][category]
            if score >= 1:
                matches.append({"ticket": ticket, "score": score})

        random.shuffle(matches)  # so that ties are broken randomly, not by the order of the file
        matches.sort(key=get_score, reverse=True)  # a stable sort: tickets with equal scores keep the random order

        log.info("%s: %d tickets contain at least one keyword", category, len(matches))
        for wanted_score in [5, 4, 3, 2, 1]:
            how_many = 0
            for match in matches:
                if match["score"] == wanted_score:
                    how_many = how_many + 1
            log.info("  with %d keyword hit(s): %d", wanted_score, how_many)

        limit = config.how_many[category]
        for match in matches[:limit]:  # [:limit] means "the first `limit` matches"
            ticket = match["ticket"]
            picked_ids.add(ticket["id"])

            row = {}
            row["id"] = ticket["id"]
            row["text"] = ticket["text"]
            row["dataset_category"] = ticket["dataset_category"]
            row["dataset_priority"] = ticket["dataset_priority"]
            row["queue"] = ticket["queue"]
            row["version"] = ticket["version"]
            row["mined_for"] = category
            picked.append(row)
        log.info("  picked %d", min(limit, len(matches)))

    # Step 4: mix the categories together, then save.
    random.shuffle(picked)
    write_jsonl(out_file, picked)
    log.info("Wrote %d tickets -> %s", len(picked), out_file)
    return picked
