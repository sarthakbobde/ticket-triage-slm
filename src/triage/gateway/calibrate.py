"""Choose the gateway's confidence threshold from data (Step 11).

The gateway sends a ticket to the teacher when the student's confidence is below a threshold. Too low
a threshold and bad answers slip through; too high and every ticket goes to the (paid) teacher.

Two steps, both on the 200 gold tickets:
  1. collect: ask the student for every ticket and record its answer AND its confidence.
  2. analyse: for each threshold, simulate the gateway. Tickets below it take the TEACHER's answer
     (already collected in Step 08, so this costs nothing); the rest keep the student's answer.
     Then score the mix against the gold labels.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from triage import paths
from triage.evals.runner import load_gold_tickets
from triage.gateway.service import ask_student
from triage.io import Row, append_row, read_done_ids, read_jsonl
from triage.llm import make_client
from triage.settings import load_endpoint

log = logging.getLogger(__name__)

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
FIELDS = ("category", "priority", "sentiment")


def collect_student_answers(config_name: str, run_name: str, runs_dir: Path = paths.EVAL_RUNS) -> Path:
    """Ask the student about every gold ticket and save answer, confidence and time. Resumable."""
    endpoint = load_endpoint(config_name)
    client = make_client(endpoint, max_retries=0, timeout=120)
    out_path = runs_dir / f"{run_name}.jsonl"
    runs_dir.mkdir(parents=True, exist_ok=True)
    done = read_done_ids(out_path, ignore_failed=True)

    todo = []
    for ticket in load_gold_tickets():
        if ticket["id"] not in done:
            todo.append(ticket)
    log.info("%s | %d gold tickets to ask (%d already done)", endpoint.model, len(todo), len(done))

    with open(out_path, "a", encoding="utf-8") as out:
        for number, ticket in enumerate(todo, 1):
            started = time.perf_counter()
            answer = ask_student(client, endpoint.model or "", ticket["text"])
            row: Row = {"id": ticket["id"], "latency_s": round(time.perf_counter() - started, 3)}
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


def _accuracy(rows: list[tuple[dict[str, Any], dict[str, Any]]], field: str) -> float:
    """Share of (answer, gold) pairs where the answer's field equals the gold's."""
    if not rows:
        return float("nan")
    right = 0
    for answer, gold in rows:
        if answer.get(field) == gold[field]:
            right = right + 1
    return right / len(rows)


def simulate(
    student: dict[str, Row], teacher: dict[str, dict[str, Any]], gold: dict[str, Row]
) -> list[dict[str, Any]]:
    """One row per threshold: how much goes to the teacher and how accurate the mix is.

    student: id -> {"answer": {...}, "confidence": float|None}   (answer None if unusable)
    teacher: id -> answer dict (the teacher's answer for that ticket)
    gold:    id -> gold label dict
    """
    table = []
    for threshold in THRESHOLDS:
        mixed = []  # (answer used, gold)
        to_teacher = 0
        for ticket_id, label in gold.items():
            entry = student[ticket_id]
            confidence = entry["confidence"]
            unsure = confidence is not None and confidence < threshold
            if entry["answer"] is None or unsure:
                to_teacher = to_teacher + 1
                mixed.append((teacher[ticket_id], label))
            else:
                mixed.append((entry["answer"], label))
        row: dict[str, Any] = {"threshold": threshold, "to_teacher": to_teacher / len(gold)}
        for field in FIELDS:
            row[field] = _accuracy(mixed, field)
        table.append(row)
    return table


def load_for_simulation(
    student_run: str, teacher_run: str
) -> tuple[dict[str, Row], dict[str, dict[str, Any]], dict[str, Row]]:
    """Read the three files the simulation needs."""
    gold = {row["id"]: row for row in read_jsonl(paths.GOLD_LABELS)}
    student: dict[str, Row] = {}
    for row in read_jsonl(paths.EVAL_RUNS / f"{student_run}.jsonl"):
        answer = json.loads(row["raw"]) if row.get("raw") else None
        student[row["id"]] = {"answer": answer, "confidence": row.get("confidence")}
    teacher = {}
    for row in read_jsonl(paths.EVAL_RUNS / f"{teacher_run}.jsonl"):
        if row.get("raw"):
            teacher[row["id"]] = json.loads(row["raw"])
    # Only compare tickets for which we have everything.
    usable = {i: g for i, g in gold.items() if i in student and i in teacher}
    return student, teacher, usable


def format_table(table: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| min confidence | sent to teacher | category acc | priority acc | sentiment acc |",
        "|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(
            f"| {row['threshold']:.2f} | {100 * row['to_teacher']:.0f}% | {100 * row['category']:.1f}% "
            f"| {100 * row['priority']:.1f}% | {100 * row['sentiment']:.1f}% |"
        )
    return lines


def analyse(student_run: str, teacher_run: str) -> list[str]:
    student, teacher, gold = load_for_simulation(student_run, teacher_run)
    return [f"Simulated on {len(gold)} gold tickets.", ""] + format_table(simulate(student, teacher, gold))
