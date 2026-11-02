"""Ask a model to triage the gold tickets and save its RAW answers (Step 05a).

Models come from configs/models.yaml (every model is reached through an OpenAI-style web address,
ADR-004). We save the raw text, not a parsed answer: the scoring step decides whether it is valid,
so a model that breaks the JSON cannot hide it.

Safe to stop and start again: ids already in the run file are skipped (failed calls are retried).
A remote server (anything not on this computer) costs money, so it needs approved=True (--yes).
"""

import logging
import time

from triage import paths
from triage.io import append_row, read_done_ids, read_jsonl
from triage.llm import make_client, require_approval_for_paid_calls, response_format_for
from triage.prompts import build_messages
from triage.settings import load_endpoint

log = logging.getLogger(__name__)


def load_gold_tickets(test_file=None):
    """The gold tickets (id and text only; the labels are NOT included)."""
    if test_file is None:
        test_file = paths.PROCESSED / "test.jsonl"
    tickets = []
    for row in read_jsonl(test_file):
        if row["gold"]:
            tickets.append({"id": row["id"], "text": row["text"]})
    return tickets


def run_model(
    config_name, run_name, model=None, fmt="none", limit=None, approved=False, client=None, runs_dir=paths.EVAL_RUNS
):
    """Ask one model about the gold tickets and save the answers. Pass a fake `client` in tests."""
    endpoint = load_endpoint(config_name)
    if model is None:
        model = endpoint.model
    if model is None:
        model = ""

    tickets = load_gold_tickets()
    if limit:
        tickets = tickets[:limit]  # [:limit] means "only the first `limit` tickets"

    out_path = runs_dir / f"{run_name}.jsonl"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Work out which tickets still need an answer. Failed ones count as not done, so they are retried.
    done = read_done_ids(out_path, ignore_failed=True)
    todo = []
    for ticket in tickets:
        if ticket["id"] not in done:
            todo.append(ticket)
    log.info("%s | %d gold tickets, %d done, %d to go", model, len(tickets), len(done), len(todo))
    if len(todo) == 0:
        return out_path

    # Money check: stop here if this would make paid calls that were not approved.
    require_approval_for_paid_calls(endpoint, len(todo), approved)
    if client is None:
        client = make_client(endpoint)
    response_format = response_format_for(fmt)

    # If the config sets a limit of requests per minute, wait long enough between calls to stay under it.
    min_gap = 0.0
    if endpoint.rpm:
        min_gap = 60 / endpoint.rpm

    number = 0
    last_call = 0.0
    with open(out_path, "a", encoding="utf-8") as out_file:
        for ticket in todo:
            number = number + 1

            # Wait if the previous call was too recent.
            wait = min_gap - (time.perf_counter() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.perf_counter()

            row = {}
            row["id"] = ticket["id"]
            row["model"] = model
            row["format"] = fmt

            started = time.perf_counter()
            try:
                messages = build_messages(ticket["text"])
                if response_format is None:
                    reply = client.chat.completions.create(model=model, messages=messages, temperature=0)
                else:
                    reply = client.chat.completions.create(
                        model=model, messages=messages, temperature=0, response_format=response_format
                    )

                answer_text = reply.choices[0].message.content
                if answer_text is None:
                    answer_text = ""
                row["raw"] = answer_text
                if reply.usage:
                    row["input_tokens"] = reply.usage.prompt_tokens
                    row["output_tokens"] = reply.usage.completion_tokens
            except Exception as error:
                # A failed call is recorded, and the scoring step counts it as wrong.
                row["raw"] = None
                row["error"] = str(error)[:300]
            row["latency_s"] = round(time.perf_counter() - started, 3)

            append_row(out_file, row)
            if number % 20 == 0 or number == len(todo):
                log.info("[%d/%d]", number, len(todo))

    log.info("Saved -> %s", out_path)
    return out_path
