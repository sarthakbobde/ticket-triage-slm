"""Scoring functions for the eval harness (Step 05).

Everything here is plain Python on lists, so you can read each metric top to bottom.
A model answer is compared with the gold label (data/gold/gold_labels.jsonl).

Rule used everywhere: an answer that is not valid JSON, or does not fit the schema, counts as
WRONG for every field. Otherwise a model could look good by only answering the easy tickets.
"""

import math
import re

from pydantic import ValidationError

from triage.schema import TicketTriage


def parse_answer(raw_text):
    """Turn the model's raw text into (TicketTriage or None, problem).

    problem is "" when fine, "not_json" when the text is not JSON at all,
    or "schema" when it is JSON but breaks the schema (wrong category word, missing field...).
    """
    if raw_text is None:
        return None, "not_json"
    try:
        answer = TicketTriage.model_validate_json(raw_text)
        return answer, ""
    except ValidationError as error:
        # Tell "not JSON" apart from "JSON with the wrong shape".
        for item in error.errors():
            if item["type"] == "json_invalid":
                return None, "not_json"
        return None, "schema"


def accuracy(true_values, predicted_values):
    """Share of tickets where the prediction equals the truth. None (invalid answer) is wrong."""
    right = 0
    for true_value, predicted_value in zip(true_values, predicted_values, strict=True):
        if true_value == predicted_value:
            right = right + 1
    return right / len(true_values)


def class_scores(true_values, predicted_values, classes):
    """Precision, recall, F1 and support for each class.

    precision = of the tickets the model put in this class, how many really belong there
    recall    = of the tickets that really belong, how many the model found
    f1        = one number combining the two (their harmonic mean)
    """
    scores = {}
    for name in classes:
        true_positive = 0  # model said name, truth is name
        false_positive = 0  # model said name, truth is something else
        false_negative = 0  # truth is name, model said something else (or gave no answer)
        for true_value, predicted_value in zip(true_values, predicted_values, strict=True):
            if predicted_value == name and true_value == name:
                true_positive = true_positive + 1
            elif predicted_value == name and true_value != name:
                false_positive = false_positive + 1
            elif predicted_value != name and true_value == name:
                false_negative = false_negative + 1

        precision = 0.0
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        recall = 0.0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        scores[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_positive + false_negative,  # how many gold tickets are in this class
        }
    return scores


def macro_f1(scores):
    """Average F1 over classes, every class counts the same however small it is."""
    total = 0.0
    count = 0
    for name in scores:
        if scores[name]["support"] == 0:
            continue  # a class with no gold tickets cannot be scored
        total = total + scores[name]["f1"]
        count = count + 1
    return total / count


def confusion(true_values, predicted_values):
    """Counts of (truth, prediction) pairs, e.g. {("billing", "customer_service"): 3}."""
    counts = {}
    for true_value, predicted_value in zip(true_values, predicted_values, strict=True):
        pair = (true_value, predicted_value)
        counts[pair] = counts.get(pair, 0) + 1
    return counts


PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2}


def priority_off_by_one(true_values, predicted_values):
    """Share of tickets where priority is wrong by exactly one step (low<->medium<->high).

    Off by one is a mild mistake; low vs high is a bad one. Accuracy alone hides that.
    """
    close = 0
    for true_value, predicted_value in zip(true_values, predicted_values, strict=True):
        if predicted_value is None:
            continue
        gap = abs(PRIORITY_RANK[true_value] - PRIORITY_RANK[predicted_value])
        if gap == 1:
            close = close + 1
    return close / len(true_values)


def clean_product_names(names):
    """Lower-case and trim, so 'Jira ' and 'jira' count as the same product."""
    cleaned = set()
    for name in names:
        text = name.strip().lower()
        if text:
            cleaned.add(text)
    return cleaned


def products_f1(true_names, predicted_names):
    """F1 between two product lists for ONE ticket. Both empty counts as a perfect 1.0."""
    true_set = clean_product_names(true_names)
    predicted_set = clean_product_names(predicted_names)
    if len(true_set) == 0 and len(predicted_set) == 0:
        return 1.0
    # How many products appear in both lists?
    both = 0
    for name in true_set:
        if name in predicted_set:
            both = both + 1
    if both == 0:
        return 0.0
    precision = both / len(predicted_set)
    recall = both / len(true_set)
    return 2 * precision * recall / (precision + recall)


def percentile(numbers, percent):
    """Nearest-rank percentile: percentile(times, 95) is the time 95% of calls beat."""
    if len(numbers) == 0:
        return None
    ordered = sorted(numbers)
    rank = math.ceil(percent / 100 * len(ordered))  # 1-based place in the sorted list
    position = min(len(ordered) - 1, max(0, rank - 1))
    return ordered[position]


# The prompt says "do not promise refunds or timelines". This is a rough check, not a judge.
PROMISE_PATTERN = re.compile(
    r"(we will refund|you will be refunded|full refund|within \d+ (hour|day|business)|"
    r"within 24|by tomorrow|guarantee)",
    re.IGNORECASE,
)


def promises_something(reply_text):
    return PROMISE_PATTERN.search(reply_text) is not None
