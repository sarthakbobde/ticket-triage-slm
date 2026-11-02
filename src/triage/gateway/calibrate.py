"""Choose the gateway's confidence threshold from data (Step 11).

The gateway sends a ticket to the teacher when the student's confidence is below a "threshold".
If the threshold is too low, bad answers slip through. If it is too high, almost every ticket goes
to the paid teacher.

There are two steps, both using the 200 gold tickets:
  1. COLLECT: ask the student about every ticket and save its answer AND its confidence.
  2. SIMULATE: for each possible threshold, pretend to be the gateway. A ticket below the threshold
     takes the TEACHER's answer (we already have those from Step 08, so this costs nothing).
     A ticket at or above it keeps the student's answer. Then score the mix against the gold labels.
"""

import json
import logging
import time

from triage import paths
from triage.evals.runner import load_gold_tickets
from triage.gateway.service import ask_student
from triage.io import append_row, read_done_ids, read_jsonl
from triage.llm import make_client
from triage.settings import load_endpoint

log = logging.getLogger(__name__)

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
FIELDS = ["category", "priority", "sentiment"]


def collect_student_answers(config_name, run_name, runs_dir=paths.EVAL_RUNS):
    """Ask the student about every gold ticket and save the answer, confidence and time.

    Safe to stop and start again: tickets that are already in the file are skipped.
    """
    endpoint = load_endpoint(config_name)
    client = make_client(endpoint, max_retries=0, timeout=120)
    out_path = runs_dir / (run_name + ".jsonl")
    runs_dir.mkdir(parents=True, exist_ok=True)
    done = read_done_ids(out_path, ignore_failed=True)

    # Make the list of tickets we still have to ask about.
    todo = []
    for ticket in load_gold_tickets():
        if ticket["id"] not in done:
            todo.append(ticket)
    log.info("%s | %d gold tickets to ask (%d already done)", endpoint.model, len(todo), len(done))

    with open(out_path, "a", encoding="utf-8") as out:
        number = 0
        for ticket in todo:
            number = number + 1
            started = time.perf_counter()
            answer = ask_student(client, endpoint.model, ticket["text"])

            row = {}
            row["id"] = ticket["id"]
            row["latency_s"] = round(time.perf_counter() - started, 3)
            if answer.result is None:
                row["raw"] = None
                row["error"] = answer.problem
            else:
                row["raw"] = answer.result.model_dump_json()
                row["confidence"] = answer.confidence
            append_row(out, row)

            if number % 20 == 0 or number == len(todo):
                log.info("[%d/%d]", number, len(todo))
    return out_path


def accuracy_of_field(pairs, field):
    """Share of (answer, gold) pairs where the answer's field equals the gold label's field."""
    if len(pairs) == 0:
        return 0.0
    right = 0
    for answer, gold in pairs:
        if answer.get(field) == gold[field]:
            right = right + 1
    return right / len(pairs)


def simulate(student, teacher, gold):
    """For each threshold: how many tickets go to the teacher, and how accurate is the mix?

    student: ticket id -> {"answer": the student's answer or None, "confidence": a number or None}
    teacher: ticket id -> the teacher's answer
    gold:    ticket id -> the gold labels

    Returns a list with one dictionary per threshold.
    """
    table = []
    for threshold in THRESHOLDS:
        pairs = []  # (the answer the gateway would give, the gold label)
        sent_to_teacher = 0

        for ticket_id in gold:
            gold_label = gold[ticket_id]
            student_entry = student[ticket_id]
            confidence = student_entry["confidence"]

            # Does this ticket go to the teacher?
            go_to_teacher = False
            if student_entry["answer"] is None:
                go_to_teacher = True  # the student gave no usable answer
            elif confidence is not None:
                if confidence < threshold:
                    go_to_teacher = True  # the student was not sure enough

            if go_to_teacher:
                sent_to_teacher = sent_to_teacher + 1
                pairs.append((teacher[ticket_id], gold_label))
            else:
                pairs.append((student_entry["answer"], gold_label))

        row = {}
        row["threshold"] = threshold
        row["to_teacher"] = sent_to_teacher / len(gold)
        for field in FIELDS:
            row[field] = accuracy_of_field(pairs, field)
        table.append(row)
    return table


def load_for_simulation(student_run, teacher_run):
    """Read the three files the simulation needs. Returns (student, teacher, gold)."""
    all_gold = {}
    for row in read_jsonl(paths.GOLD_LABELS):
        all_gold[row["id"]] = row

    student = {}
    for row in read_jsonl(paths.EVAL_RUNS / (student_run + ".jsonl")):
        answer = None
        if row.get("raw"):
            answer = json.loads(row["raw"])
        student[row["id"]] = {"answer": answer, "confidence": row.get("confidence")}

    teacher = {}
    for row in read_jsonl(paths.EVAL_RUNS / (teacher_run + ".jsonl")):
        if row.get("raw"):
            teacher[row["id"]] = json.loads(row["raw"])

    # Only compare tickets for which we have the student, the teacher AND the gold label.
    gold = {}
    for ticket_id in all_gold:
        if ticket_id in student and ticket_id in teacher:
            gold[ticket_id] = all_gold[ticket_id]
    return student, teacher, gold


def format_table(table):
    """Turn the simulation rows into a markdown table (a list of text lines)."""
    lines = []
    lines.append("| min confidence | sent to teacher | category acc | priority acc | sentiment acc |")
    lines.append("|---|---|---|---|---|")
    for row in table:
        lines.append(
            "| %.2f | %.0f%% | %.1f%% | %.1f%% | %.1f%% |"
            % (
                row["threshold"],
                100 * row["to_teacher"],
                100 * row["category"],
                100 * row["priority"],
                100 * row["sentiment"],
            )
        )
    return lines


def analyse(student_run, teacher_run):
    """Run the simulation on saved files and return the report as a list of text lines."""
    student, teacher, gold = load_for_simulation(student_run, teacher_run)
    table = simulate(student, teacher, gold)

    lines = []
    lines.append("Simulated on %d gold tickets." % len(gold))
    lines.append("")
    for line in format_table(table):
        lines.append(line)
    return lines
