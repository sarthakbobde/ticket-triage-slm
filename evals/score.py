"""Step 05b: score a run against the gold labels. Free: no model is called.

  uv run python evals/score.py --run teacher-gpt-4.1

Reads  evals/runs/<run>.jsonl            (raw answers from evals/run_model.py)
       data/gold/gold_labels.jsonl       (the 200 reviewed labels)
Writes evals/reports/<run>.md and evals/reports/<run>.json (the numbers, for comparing runs)

An invalid answer counts as WRONG for every field (see src/triage/metrics.py).
"""

import argparse
import json
import sys
from pathlib import Path

from triage import metrics
from triage.schema import Category, Priority, Sentiment

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
GOLD_FILE = ROOT / "data" / "gold" / "gold_labels.jsonl"
RUNS_DIR = ROOT / "evals" / "runs"
REPORTS_DIR = ROOT / "evals" / "reports"


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percent(number):
    return f"{100 * number:.1f}%"


def field_report(classes, true_values, predicted_values):
    """Numbers for one classification field (category, priority or sentiment)."""
    scores = metrics.class_scores(true_values, predicted_values, classes)
    return {
        "accuracy": metrics.accuracy(true_values, predicted_values),
        "macro_f1": metrics.macro_f1(scores),
        "classes": scores,
        "confusion": metrics.confusion(true_values, predicted_values),
    }


def class_table(report, classes):
    lines = ["| class | precision | recall | F1 | gold tickets |", "|---|---|---|---|---|"]
    for name in classes:
        s = report["classes"][name]
        lines.append(
            f"| {name} | {percent(s['precision'])} | {percent(s['recall'])} | "
            f"{percent(s['f1'])} | {s['support']} |"
        )
    return lines


def confusion_table(report, classes):
    """Rows = what the gold says, columns = what the model said ('invalid' = no usable answer)."""
    columns = list(classes) + ["invalid"]
    lines = ["| gold \\ model | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for true_name in classes:
        cells = []
        for predicted_name in columns:
            key = predicted_name if predicted_name != "invalid" else None
            cells.append(str(report["confusion"].get((true_name, key), 0)))
        lines.append(f"| {true_name} | " + " | ".join(cells) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    # A retried ticket appears twice in the file (failed call, then the retry): keep the last.
    latest = {}
    for row in read_jsonl(RUNS_DIR / f"{args.run}.jsonl"):
        latest[row["id"]] = row
    run_rows = list(latest.values())
    gold = {}
    for row in read_jsonl(GOLD_FILE):
        gold[row["id"]] = row

    # Line up each answer with its gold label.
    true_category, true_priority, true_sentiment = [], [], []
    pred_category, pred_priority, pred_sentiment = [], [], []
    product_scores = []
    problems = {"not_json": 0, "schema": 0, "": 0}
    latencies = []
    output_tokens = []
    replies_valid = 0
    replies_with_promise = 0

    for row in run_rows:
        label = gold[row["id"]]
        answer, problem = metrics.parse_answer(row.get("raw"))
        problems[problem] = problems[problem] + 1
        if row.get("latency_s") is not None:
            latencies.append(row["latency_s"])
        if row.get("output_tokens") is not None:
            output_tokens.append(row["output_tokens"])

        true_category.append(label["category"])
        true_priority.append(label["priority"])
        true_sentiment.append(label["sentiment"])
        if answer is None:
            pred_category.append(None)
            pred_priority.append(None)
            pred_sentiment.append(None)
            product_scores.append(0.0)
        else:
            pred_category.append(answer.category.value)
            pred_priority.append(answer.priority.value)
            pred_sentiment.append(answer.sentiment.value)
            product_scores.append(metrics.products_f1(label["products"], answer.products))
            replies_valid = replies_valid + 1
            if metrics.promises_something(answer.suggested_reply):
                replies_with_promise = replies_with_promise + 1

    count = len(run_rows)
    category_classes = [c.value for c in Category]
    priority_classes = [p.value for p in Priority]
    sentiment_classes = [s.value for s in Sentiment]
    category = field_report(category_classes, true_category, pred_category)
    priority = field_report(priority_classes, true_priority, pred_priority)
    sentiment = field_report(sentiment_classes, true_sentiment, pred_sentiment)

    summary = {
        "run": args.run,
        "tickets": count,
        "json_valid": 1 - problems["not_json"] / count,
        "schema_valid": problems[""] / count,
        "category_accuracy": category["accuracy"],
        "category_macro_f1": category["macro_f1"],
        "priority_accuracy": priority["accuracy"],
        "priority_off_by_one": metrics.priority_off_by_one(true_priority, pred_priority),
        "sentiment_accuracy": sentiment["accuracy"],
        "products_f1": sum(product_scores) / count,
        "reply_promise_rate": (replies_with_promise / replies_valid) if replies_valid else None,
        "latency_p50_s": metrics.percentile(latencies, 50),
        "latency_p95_s": metrics.percentile(latencies, 95),
        "avg_output_tokens": (sum(output_tokens) / len(output_tokens)) if output_tokens else None,
    }

    lines = [f"# Eval: {args.run}", "", f"{count} gold tickets. Invalid answers count as wrong.", ""]
    lines += [
        "## Headline",
        "",
        "| metric | value |",
        "|---|---|",
        f"| valid JSON | {percent(summary['json_valid'])} |",
        f"| fits the schema | {percent(summary['schema_valid'])} |",
        f"| category accuracy | {percent(summary['category_accuracy'])} |",
        f"| category macro-F1 | {percent(summary['category_macro_f1'])} |",
        f"| priority accuracy | {percent(summary['priority_accuracy'])} |",
        f"| priority off by one step | {percent(summary['priority_off_by_one'])} |",
        f"| sentiment accuracy | {percent(summary['sentiment_accuracy'])} |",
        f"| products F1 (per ticket, averaged) | {percent(summary['products_f1'])} |",
    ]
    if summary["reply_promise_rate"] is not None:
        lines.append(f"| replies that promise refunds or timelines | {percent(summary['reply_promise_rate'])} |")
    if summary["latency_p50_s"] is not None:
        lines.append(f"| latency p50 / p95 | {summary['latency_p50_s']:.2f}s / {summary['latency_p95_s']:.2f}s |")
    lines += [
        "",
        f"Invalid answers: {problems['not_json']} not JSON, {problems['schema']} JSON that breaks the schema.",
        "",
        "## Category per class",
        "",
    ]
    lines += class_table(category, category_classes)
    lines += ["", "Confusion (rows = gold, columns = model):", ""]
    lines += confusion_table(category, category_classes)
    lines += ["", "## Priority per class", ""]
    lines += class_table(priority, priority_classes)
    lines += ["", "## Sentiment per class", ""]
    lines += class_table(sentiment, sentiment_classes)
    lines += [
        "",
        "Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{args.run}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS_DIR / f"{args.run}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved -> evals/reports/{args.run}.md")


if __name__ == "__main__":
    main()
