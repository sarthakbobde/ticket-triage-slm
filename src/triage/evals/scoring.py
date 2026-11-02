"""Score a run against the gold labels (Step 05b). Free: no model is called.

Reads  evals/runs/<run>.jsonl          the raw answers (from triage.evals.runner or the Kaggle notebook)
       data/gold/gold_labels.jsonl     the 200 reviewed labels
Writes evals/reports/<run>.md and .json (the numbers, so runs can be compared)

An invalid answer counts as WRONG for every field (see metrics.py).
"""

import json
import logging

from triage import paths
from triage.evals import metrics
from triage.io import read_jsonl
from triage.schema import Category, Priority, Sentiment

log = logging.getLogger(__name__)


def percent(number):
    """0.795 becomes the text '79.5%'."""
    return f"{100 * number:.1f}%"


def add_lines(lines, new_lines):
    """Add every line of `new_lines` to the end of `lines`."""
    for line in new_lines:
        lines.append(line)


def field_report(classes, true_values, predicted_values):
    """All the numbers for one field (category, priority or sentiment)."""
    scores = metrics.class_scores(true_values, predicted_values, classes)
    report = {}
    report["accuracy"] = metrics.accuracy(true_values, predicted_values)
    report["macro_f1"] = metrics.macro_f1(scores)
    report["classes"] = scores
    report["confusion"] = metrics.confusion(true_values, predicted_values)
    return report


def class_table(report, classes):
    """A markdown table with precision, recall and F1 for each class."""
    lines = []
    lines.append("| class | precision | recall | F1 | gold tickets |")
    lines.append("|---|---|---|---|---|")
    for name in classes:
        s = report["classes"][name]
        lines.append(
            f"| {name} | {percent(s['precision'])} | {percent(s['recall'])} | {percent(s['f1'])} | {s['support']} |"
        )
    return lines


def confusion_table(report, classes):
    """A markdown table. Rows = what the gold says, columns = what the model said.

    The last column, 'invalid', counts answers that could not be used at all.
    """
    columns = list(classes)
    columns.append("invalid")

    lines = []
    lines.append("| gold \\ model | " + " | ".join(columns) + " |")
    lines.append("|---" * (len(columns) + 1) + "|")
    for true_name in classes:
        cells = []
        for predicted_name in columns:
            # An invalid answer is stored as None in the confusion counts.
            key = predicted_name
            if predicted_name == "invalid":
                key = None
            count = report["confusion"].get((true_name, key), 0)
            cells.append(str(count))
        lines.append(f"| {true_name} | " + " | ".join(cells) + " |")
    return lines


def latest_row_per_ticket(rows):
    """Keep one row per ticket.

    A retried ticket appears twice in a run file (the failed call, then the retry). We keep the last.
    """
    latest = {}
    for row in rows:
        latest[row["id"]] = row
    return list(latest.values())


def score_run(run_name, runs_dir=paths.EVAL_RUNS, reports_dir=paths.EVAL_REPORTS, gold_file=paths.GOLD_LABELS):
    """Score one run. Writes the report files and returns (the summary numbers, the report lines)."""
    run_rows = latest_row_per_ticket(read_jsonl(runs_dir / f"{run_name}.jsonl"))

    gold = {}
    for row in read_jsonl(gold_file):
        gold[row["id"]] = row

    # Line up each answer with its gold label: one list of true values and one of the model's values.
    true_category = []
    true_priority = []
    true_sentiment = []
    pred_category = []
    pred_priority = []
    pred_sentiment = []
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
            # No usable answer: wrong for every field.
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

    # The list of allowed values for each field.
    category_classes = []
    for category_value in Category:
        category_classes.append(category_value.value)
    priority_classes = []
    for priority_value in Priority:
        priority_classes.append(priority_value.value)
    sentiment_classes = []
    for sentiment_value in Sentiment:
        sentiment_classes.append(sentiment_value.value)

    category = field_report(category_classes, true_category, pred_category)
    priority = field_report(priority_classes, true_priority, pred_priority)
    sentiment = field_report(sentiment_classes, true_sentiment, pred_sentiment)

    # Numbers that only exist if there was something to measure.
    reply_promise_rate = None
    if replies_valid > 0:
        reply_promise_rate = replies_with_promise / replies_valid
    avg_output_tokens = None
    if len(output_tokens) > 0:
        avg_output_tokens = sum(output_tokens) / len(output_tokens)

    summary = {}
    summary["run"] = run_name
    summary["tickets"] = count
    summary["json_valid"] = 1 - problems["not_json"] / count
    summary["schema_valid"] = problems[""] / count
    summary["category_accuracy"] = category["accuracy"]
    summary["category_macro_f1"] = category["macro_f1"]
    summary["priority_accuracy"] = priority["accuracy"]
    summary["priority_off_by_one"] = metrics.priority_off_by_one(true_priority, pred_priority)
    summary["sentiment_accuracy"] = sentiment["accuracy"]
    summary["products_f1"] = sum(product_scores) / count
    summary["reply_promise_rate"] = reply_promise_rate
    summary["latency_p50_s"] = metrics.percentile(latencies, 50)
    summary["latency_p95_s"] = metrics.percentile(latencies, 95)
    summary["avg_output_tokens"] = avg_output_tokens

    # Now write the report as markdown text.
    lines = []
    lines.append(f"# Eval: {run_name}")
    lines.append("")
    lines.append(f"{count} gold tickets. Invalid answers count as wrong.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| valid JSON | {percent(summary['json_valid'])} |")
    lines.append(f"| fits the schema | {percent(summary['schema_valid'])} |")
    lines.append(f"| category accuracy | {percent(summary['category_accuracy'])} |")
    lines.append(f"| category macro-F1 | {percent(summary['category_macro_f1'])} |")
    lines.append(f"| priority accuracy | {percent(summary['priority_accuracy'])} |")
    lines.append(f"| priority off by one step | {percent(summary['priority_off_by_one'])} |")
    lines.append(f"| sentiment accuracy | {percent(summary['sentiment_accuracy'])} |")
    lines.append(f"| products F1 (per ticket, averaged) | {percent(summary['products_f1'])} |")
    if summary["reply_promise_rate"] is not None:
        lines.append(f"| replies that promise refunds or timelines | {percent(summary['reply_promise_rate'])} |")
    if summary["latency_p50_s"] is not None:
        lines.append(f"| latency p50 / p95 | {summary['latency_p50_s']:.2f}s / {summary['latency_p95_s']:.2f}s |")
    lines.append("")
    lines.append(f"Invalid answers: {problems['not_json']} not JSON, {problems['schema']} JSON that breaks the schema.")
    lines.append("")
    lines.append("## Category per class")
    lines.append("")
    add_lines(lines, class_table(category, category_classes))
    lines.append("")
    lines.append("Confusion (rows = gold, columns = model):")
    lines.append("")
    add_lines(lines, confusion_table(category, category_classes))
    lines.append("")
    lines.append("## Priority per class")
    lines.append("")
    add_lines(lines, class_table(priority, priority_classes))
    lines.append("")
    lines.append("## Sentiment per class")
    lines.append("")
    add_lines(lines, class_table(sentiment, sentiment_classes))
    lines.append("")
    lines.append("Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.")

    # Save the report (markdown) and the numbers (json).
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{run_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports_dir / f"{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Saved -> %s", reports_dir / f"{run_name}.md")
    return summary, lines
