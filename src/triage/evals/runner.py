"""Ask a model to triage the gold tickets and save its RAW answers (Step 05a).

Models come from configs/models.yaml (every model is an OpenAI-compatible endpoint, ADR-004).
We save the raw text, not a parsed answer: scoring decides whether it is valid, so a model that
breaks the JSON cannot hide it.

Resumable: ids already in the run file are skipped (failed calls are retried).
Remote endpoints (anything not on localhost) cost money, so they need approved=True (--yes).
"""

import logging
import time
from pathlib import Path

from openai import OpenAI

from triage import paths
from triage.io import append_row, read_done_ids, read_jsonl
from triage.llm import make_client, require_approval_for_paid_calls, response_format_for
from triage.prompts import build_messages
from triage.settings import load_endpoint

log = logging.getLogger(__name__)


def load_gold_tickets(test_file: Path | None = None) -> list[dict[str, str]]:
    test_file = test_file or paths.PROCESSED / "test.jsonl"
    tickets = []
    for row in read_jsonl(test_file):
        if row["gold"]:
            tickets.append({"id": row["id"], "text": row["text"]})
    return tickets


def run_model(
    config_name: str,
    run_name: str,
    model: str | None = None,
    fmt: str = "none",
    limit: int | None = None,
    approved: bool = False,
    client: OpenAI | None = None,
    runs_dir: Path = paths.EVAL_RUNS,
) -> Path:
    """Answer the gold tickets with one model. Pass `client` to use a fake in tests."""
    endpoint = load_endpoint(config_name)
    model = model or endpoint.model or ""
    tickets = load_gold_tickets()
    if limit:
        tickets = tickets[:limit]

    out_path = runs_dir / f"{run_name}.jsonl"
    runs_dir.mkdir(parents=True, exist_ok=True)
    done = read_done_ids(out_path, ignore_failed=True)
    todo = []
    for ticket in tickets:
        if ticket["id"] not in done:
            todo.append(ticket)
    log.info("%s | %d gold tickets, %d done, %d to go", model, len(tickets), len(done), len(todo))
    if not todo:
        return out_path

    require_approval_for_paid_calls(endpoint, len(todo), approved)
    client = client or make_client(endpoint)
    response_format = response_format_for(fmt)
    min_gap = 0.0
    if endpoint.rpm:
        min_gap = 60 / endpoint.rpm

    last_call = 0.0
    with open(out_path, "a", encoding="utf-8") as out:
        for number, ticket in enumerate(todo, 1):
            wait = min_gap - (time.perf_counter() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.perf_counter()

            row = {"id": ticket["id"], "model": model, "format": fmt}
            started = time.perf_counter()
            try:
                request = {"model": model, "messages": build_messages(ticket["text"]), "temperature": 0}
                if response_format is not None:
                    request["response_format"] = response_format
                reply = client.chat.completions.create(**request)
                row["raw"] = reply.choices[0].message.content or ""
                if reply.usage:
                    row["input_tokens"] = reply.usage.prompt_tokens
                    row["output_tokens"] = reply.usage.completion_tokens
            except Exception as error:  # noqa: BLE001 - keep going: a failed call is scored as a failure
                row["raw"] = None
                row["error"] = str(error)[:300]
            row["latency_s"] = round(time.perf_counter() - started, 3)

            append_row(out, row)
            if number % 20 == 0 or number == len(todo):
                log.info("[%d/%d]", number, len(todo))

    log.info("Saved -> %s", out_path)
    return out_path
