"""Step 03c: find extra tickets that probably belong to the rare categories (hr, customer_service).

Why: the teacher found only ~1% hr and ~2% customer_service in the first 1,000 training tickets
(see docs/steps/03-teacher-labels.md). That is too few for the small model to learn from, so we go
looking for more of them in tickets we have not used yet.

How: we give every unused ticket a score = how many hr (or customer-service) words it contains,
then keep the highest scorers. This is only a way to PICK tickets to label. The teacher still
decides the real category, and many picked tickets will turn out to be something else.

There are two rounds (change ROUND below, then run):
  Round 1: broad word lists for hr and customer_service  -> train_extra.jsonl   (already done)
  Round 2: strict hr word list only, built from the 34 tickets the teacher
           confirmed as hr in round 1                     -> train_extra2.jsonl
Do not run round 1 again: it would skip everything it picked the first time.

Run:  uv run python scripts/03_mine_rare_classes.py
Then: uv run python scripts/03_label_with_teacher.py --split train_extra2

Never touched: test.jsonl and val.jsonl (those stay natural and frozen).
"""

import json
import random
import sys
from pathlib import Path

from triage.dataset_map import QUEUE_TO_CATEGORY
from triage.textclean import clean_text, normalise

sys.stdout.reconfigure(encoding="utf-8")

ROUND = 2
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "tickets_en.jsonl"
PROCESSED = ROOT / "data" / "processed"
LABELED = ROOT / "data" / "labeled"

# Single words must match a WHOLE word ("hr" must not match "three").
# Phrases only need to appear somewhere in the text.
BROAD_KEYWORDS = {
    "hr": {
        "words": [
            "hr", "payroll", "salary", "salaries", "paycheck", "paychecks", "onboarding",
            "offboarding", "recruitment", "recruiting", "hiring", "vacation", "timesheet",
            "timesheets", "benefits", "bonus", "promotion", "resignation", "termination",
            "workplace", "employee", "employees", "staff", "manager", "colleague", "payslip",
        ],
        "phrases": [
            "human resources", "performance review", "sick leave", "parental leave",
            "paid time off", "work schedule", "team morale", "job offer", "time off",
        ],
    },
    "customer_service": {
        "words": [
            "return", "returns", "returned", "exchange", "exchanges", "order", "orders",
            "delivery", "shipping", "shipment", "pricing", "quote", "demo", "cancel",
            "cancellation", "complaint", "warranty", "replacement", "inquiry", "inquire",
            "enquiry", "brochure", "catalog", "catalogue",
        ],
        "phrases": [
            "pricing plans", "product information", "services offered", "more information",
            "change my", "update my", "account details", "customer feedback", "free trial",
        ],
    },
}

# Round 2: the themes that showed up in the 34 teacher-confirmed hr tickets.
# Generic words like "employee" and "staff" are left out on purpose: they appear in
# ordinary product questions, which is why round 1 had a 10% hit rate for hr.
STRICT_HR_KEYWORDS = {
    "hr": {
        "words": [
            "hr", "payroll", "onboarding", "offboarding", "turnover", "payslip", "payslips",
            "paycheck", "paychecks", "recruitment", "hiring", "timesheet", "timesheets",
            "vacation", "salary", "salaries", "resignation", "restructuring",
        ],
        "phrases": [
            "human resources", "employee training", "employee engagement", "employee benefits",
            "employee turnover", "employee productivity", "employee departure",
            "employee onboarding", "remote work", "flexible scheduling", "work schedule",
            "time off", "sick leave", "parental leave", "performance review",
        ],
    },
}

if ROUND == 1:
    KEYWORDS = BROAD_KEYWORDS
    HOW_MANY = {"hr": 200, "customer_service": 300}  # candidates to pick per category
    OUT_FILE = PROCESSED / "train_extra.jsonl"
else:
    KEYWORDS = STRICT_HR_KEYWORDS
    HOW_MANY = {"hr": 150}
    OUT_FILE = PROCESSED / "train_extra2.jsonl"


def read_jsonl(path):
    """Read a .jsonl file (one JSON object per line) into a list of dictionaries."""
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def split_into_words(text):
    """'Hello, World!' -> ['hello', 'world']. Letters and digits stay, everything else is a gap."""
    lower_text = text.lower()
    cleaned = ""
    for character in lower_text:
        if character.isalnum():
            cleaned = cleaned + character
        else:
            cleaned = cleaned + " "
    return cleaned.split()


def keyword_score(lower_text, ticket_words, keywords):
    """How many different keywords from the list does this ticket contain?"""
    score = 0
    for word in keywords["words"]:
        if word in ticket_words:
            score = score + 1
    for phrase in keywords["phrases"]:
        if phrase in lower_text:
            score = score + 1
    return score


def get_score(candidate):
    """Used for sorting: the sort looks at this number for each candidate."""
    return candidate["score"]


def main():
    # 1. Which tickets are already used? Test and val are off limits for ever,
    #    and every training ticket we already paid to label is done.
    used_ids = set()
    already_used_files = [
        PROCESSED / "test.jsonl",
        PROCESSED / "val.jsonl",
        LABELED / "train__gpt-4.1.jsonl",
        LABELED / "train_extra__gpt-4.1.jsonl",
    ]
    for path in already_used_files:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            used_ids.add(row["id"])

    # Texts of used tickets (from the processed files, which hold the cleaned text),
    # so we also skip tickets that are near-copies of ones we already used.
    used_texts = set()
    for name in ["test.jsonl", "val.jsonl", "train.jsonl", "train_extra.jsonl"]:
        path = PROCESSED / name
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row["id"] in used_ids:
                used_texts.add(normalise(row["text"]))
    print(f"Already used: {len(used_ids)} tickets")

    # 2. Score every unused ticket for each category we are mining.
    candidates = []
    seen_texts = set()
    for ticket in read_jsonl(RAW_FILE):
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
        for category in KEYWORDS:
            scores[category] = keyword_score(lower_text, ticket_words, KEYWORDS[category])
        candidates.append({
            "id": ticket["id"],
            "text": text,
            "dataset_category": QUEUE_TO_CATEGORY[ticket["queue"]].value,
            "dataset_priority": ticket["priority"],
            "queue": ticket["queue"],
            "version": ticket["version"],
            "scores": scores,
        })
    print(f"Unused tickets to search: {len(candidates)}")

    # 3. For each category: keep tickets with at least one hit, best scores first.
    random.seed(SEED)
    picked_ids = set()
    picked = []
    for category in KEYWORDS:  # hr comes first: it is the rarest
        matches = []
        for ticket in candidates:
            if ticket["id"] in picked_ids:
                continue  # already picked for the other category
            score = ticket["scores"][category]
            if score >= 1:
                matches.append({"ticket": ticket, "score": score})

        random.shuffle(matches)  # so ties are broken randomly, not by file order
        matches.sort(key=get_score, reverse=True)  # stable sort: ties keep the random order

        print(f"\n{category}: {len(matches)} tickets contain at least one keyword")
        for wanted_score in [5, 4, 3, 2, 1]:
            how_many = 0
            for match in matches:
                if match["score"] == wanted_score:
                    how_many = how_many + 1
            print(f"  with {wanted_score} keyword hit(s): {how_many}")

        limit = HOW_MANY[category]
        for match in matches[:limit]:
            ticket = match["ticket"]
            picked_ids.add(ticket["id"])
            picked.append({
                "id": ticket["id"],
                "text": ticket["text"],
                "dataset_category": ticket["dataset_category"],
                "dataset_priority": ticket["dataset_priority"],
                "queue": ticket["queue"],
                "version": ticket["version"],
                "mined_for": category,
            })
        print(f"  picked {min(limit, len(matches))}")

    # 4. Shuffle so the categories are mixed, then save.
    random.shuffle(picked)
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for row in picked:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(picked)} tickets -> {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
