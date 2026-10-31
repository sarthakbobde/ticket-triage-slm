"""Checks for src/triage/metrics.py. Run:  uv run pytest

Our plain-loop metrics are compared with scikit-learn's, which are the standard reference.
"""

import json

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from triage import metrics

CLASSES = ["technical", "product", "billing"]
TRUE = ["technical", "technical", "product", "product", "billing", "billing", "technical", "product"]
PRED = ["technical", "product", "product", "technical", "billing", "product", "technical", "product"]


def test_accuracy_matches_sklearn():
    assert metrics.accuracy(TRUE, PRED) == accuracy_score(TRUE, PRED)


def test_class_scores_match_sklearn():
    ours = metrics.class_scores(TRUE, PRED, CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(TRUE, PRED, labels=CLASSES, zero_division=0)
    for i, name in enumerate(CLASSES):
        assert abs(ours[name]["precision"] - precision[i]) < 1e-9
        assert abs(ours[name]["recall"] - recall[i]) < 1e-9
        assert abs(ours[name]["f1"] - f1[i]) < 1e-9
        assert ours[name]["support"] == support[i]


def test_macro_f1_matches_sklearn():
    ours = metrics.macro_f1(metrics.class_scores(TRUE, PRED, CLASSES))
    assert abs(ours - f1_score(TRUE, PRED, labels=CLASSES, average="macro", zero_division=0)) < 1e-9


def test_invalid_answer_counts_as_wrong_everywhere():
    true = ["technical", "product"]
    pred = ["technical", None]  # second answer was invalid
    assert metrics.accuracy(true, pred) == 0.5
    scores = metrics.class_scores(true, pred, CLASSES)
    assert scores["product"]["recall"] == 0.0  # the product ticket was missed


def test_macro_f1_skips_classes_with_no_gold_tickets():
    scores = metrics.class_scores(["technical"], ["technical"], CLASSES)
    assert metrics.macro_f1(scores) == 1.0


def test_priority_off_by_one():
    true = ["low", "low", "high", "medium"]
    pred = ["medium", "high", "high", None]
    # low->medium is one step; low->high is two steps; high->high is right; None is skipped
    assert metrics.priority_off_by_one(true, pred) == 0.25


def test_products_f1():
    assert metrics.products_f1([], []) == 1.0
    assert metrics.products_f1(["Jira"], []) == 0.0
    assert metrics.products_f1(["Jira "], ["jira"]) == 1.0  # case and spaces do not matter
    assert abs(metrics.products_f1(["Jira", "Slack"], ["Jira"]) - 2 / 3) < 1e-9


def test_percentile():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert metrics.percentile(numbers, 50) == 5
    assert metrics.percentile(numbers, 95) == 10
    assert metrics.percentile([], 50) is None


def test_parse_answer_kinds():
    good = {
        "category": "billing", "priority": "low", "sentiment": "neutral",
        "products": [], "summary": "s", "suggested_reply": "r",
    }
    answer, problem = metrics.parse_answer(json.dumps(good))
    assert problem == "" and answer.category.value == "billing"
    assert metrics.parse_answer("Sure! Here is the JSON")[1] == "not_json"
    assert metrics.parse_answer(None)[1] == "not_json"
    good["category"] = "hr"  # removed in schema v1.1
    assert metrics.parse_answer(json.dumps(good))[1] == "schema"


def test_promise_detector():
    assert metrics.promises_something("We will refund you in full.")
    assert metrics.promises_something("This will be fixed within 24 hours.")
    assert not metrics.promises_something("Thanks, we are looking into it.")
