"""Build the hand-labelling sheet for the 200 gold tickets (Step 04).

The sheet is one HTML file. It shows ONLY the ticket text, never the dataset's or the teacher's
labels, so nothing can anchor the person labelling. Answers are saved in the browser, and
"Export" downloads them as gold_labels.jsonl.
"""

import json
import logging
import random
from pathlib import Path

from triage import paths
from triage.io import read_jsonl
from triage.prompts import CATEGORY_DEFINITIONS
from triage.schema import Category, Sentiment

log = logging.getLogger(__name__)

SEED = 42
TEMPLATE = Path(__file__).parent / "templates" / "gold_sheet.html"
OUT_FILE = paths.GOLD / "label_gold.html"

# Same rubric as SYSTEM_PROMPT in prompts.py (keep the two in sync).
PRIORITY_RULES = {
    "high": "customer blocked, no workaround",
    "medium": "problem with a workaround, or time-sensitive question",
    "low": "general question, feedback, feature request",
}


def build_gold_sheet(test_file: Path = paths.PROCESSED / "test.jsonl", out_file: Path = OUT_FILE) -> int:
    """Write the sheet and return how many tickets it holds."""
    # 1. The gold tickets, text only.
    gold = []
    for row in read_jsonl(test_file):
        if row["gold"]:
            gold.append({"id": row["id"], "text": row["text"]})

    # 2. Mix the order (file order may follow the dataset's queues, which could hint at labels).
    random.seed(SEED)
    random.shuffle(gold)

    # 3. Fill the template.
    categories = []
    category_rules = []
    for category in Category:
        categories.append(category.value)
        category_rules.append([category.value, CATEGORY_DEFINITIONS[category]])
    priority_rules = []
    for name in ["high", "medium", "low"]:
        priority_rules.append([name, PRIORITY_RULES[name]])
    sentiments = []
    for sentiment in Sentiment:
        sentiments.append(sentiment.value)

    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("__TICKETS__", json.dumps(gold, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__CATEGORIES__", json.dumps(categories))
    page = page.replace("__PRIORITIES__", json.dumps(["low", "medium", "high"]))
    page = page.replace("__SENTIMENTS__", json.dumps(sentiments))
    page = page.replace("__CAT_RULES__", json.dumps(category_rules))
    page = page.replace("__PRI_RULES__", json.dumps(priority_rules))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(page, encoding="utf-8")
    log.info("%d gold tickets -> %s", len(gold), out_file)
    return len(gold)
