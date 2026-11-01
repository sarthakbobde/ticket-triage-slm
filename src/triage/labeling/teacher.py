"""Label tickets with the teacher model -> data/labeled/<split>__<model>.jsonl (Step 03).

- Same SYSTEM_PROMPT as training and serving (prompts.py).
- Every answer is validated with TicketTriage; invalid -> one retry -> failures file.
- Resumable: ids already in the output file are skipped, so a crash never costs money twice.
- Every row records SCHEMA_VERSION, the teacher model and token usage (for the cost report).
- Calls cost money: a remote endpoint needs approved=True (the --yes flag on the command line).
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from triage import paths
from triage.errors import UserError
from triage.io import append_row, read_done_ids, read_jsonl
from triage.llm import make_client, require_approval_for_paid_calls, response_format_for
from triage.prompts import build_messages
from triage.schema import SCHEMA_VERSION, TicketTriage
from triage.settings import ModelEndpoint, load_endpoint

log = logging.getLogger(__name__)

LABEL_SPLITS = ["val", "train", "train_extra", "train_extra2", "test"]


@dataclass
class LabelRunResult:
    labelled: int
    failed: int
    out_path: Path


def label_one(
    client: OpenAI, endpoint: ModelEndpoint, model: str, text: str
) -> tuple[TicketTriage, dict[str, Any]]:
    """One ticket -> validated label plus call details. Retries once on invalid JSON, then raises."""
    extra = {}
    if endpoint.reasoning_effort:
        extra["reasoning_effort"] = endpoint.reasoning_effort
    last_error = ""
    for _attempt in range(2):
        started = time.perf_counter()
        reply = client.chat.completions.create(
            model=model,
            messages=build_messages(text),
            temperature=endpoint.temperature,
            response_format=response_format_for(endpoint.response_format),
            **extra,
        )
        details = {
            "latency_s": round(time.perf_counter() - started, 2),
            "input_tokens": reply.usage.prompt_tokens if reply.usage else None,
            "output_tokens": reply.usage.completion_tokens if reply.usage else None,
        }
        raw = reply.choices[0].message.content or ""
        try:
            return TicketTriage.model_validate_json(raw), details
        except ValidationError as error:
            last_error = f"{error.error_count()} errors: {error.errors()[0]['msg']} | raw={raw[:300]}"
    raise ValueError(last_error)


def list_models(client: OpenAI) -> list[str]:
    """Model ids this key can use."""
    return sorted(model.id for model in client.models.list())


def label_split(
    split: str,
    model: str | None = None,
    limit: int | None = None,
    approved: bool = False,
    endpoint: ModelEndpoint | None = None,
    client: OpenAI | None = None,
) -> LabelRunResult:
    """Label one split. Pass `client` to use a fake in tests."""
    if split not in LABEL_SPLITS:
        raise UserError(f"Unknown split '{split}'. Choose from {LABEL_SPLITS}.")
    endpoint = endpoint or load_endpoint("teacher")
    model = model or endpoint.model
    if not model:
        raise UserError("No model: pass --model or set teacher.model in configs/models.yaml.")

    tickets = read_jsonl(paths.PROCESSED / f"{split}.jsonl")
    if limit:
        tickets = tickets[:limit]

    safe_model = model.replace("/", "_")
    out_path = paths.LABELED / f"{split}__{safe_model}.jsonl"
    fail_path = paths.LABELED / f"failures__{safe_model}.jsonl"
    paths.LABELED.mkdir(parents=True, exist_ok=True)
    done = read_done_ids(out_path)
    todo = []
    for ticket in tickets:
        if ticket["id"] not in done:
            todo.append(ticket)
    log.info(
        "%s | %s: %d tickets, %d already labelled, %d to go", model, split, len(tickets), len(done), len(todo)
    )
    if not todo:
        return LabelRunResult(0, 0, out_path)

    require_approval_for_paid_calls(endpoint, len(todo), approved)
    client = client or make_client(endpoint, max_retries=6, timeout=120)

    min_gap = 0.0  # stay under the deployment's rate limit
    if endpoint.rpm:
        min_gap = 60 / endpoint.rpm
    n_ok = 0
    n_fail = 0
    last_call = 0.0
    with open(out_path, "a", encoding="utf-8") as out, open(fail_path, "a", encoding="utf-8") as fail:
        for number, ticket in enumerate(todo, 1):
            wait = min_gap - (time.perf_counter() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.perf_counter()
            try:
                label, details = label_one(client, endpoint, model, ticket["text"])
            except Exception as error:  # noqa: BLE001 - log and move on; never lose the whole run
                n_fail = n_fail + 1
                append_row(fail, {"id": ticket["id"], "split": split, "error": str(error)[:500]})
                log.warning("[%d/%d] %s FAILED: %s", number, len(todo), ticket["id"], str(error)[:120])
                continue
            n_ok = n_ok + 1
            append_row(
                out,
                {
                    "id": ticket["id"],
                    "schema_version": SCHEMA_VERSION,
                    "teacher": model,
                    "label": label.model_dump(mode="json"),
                    **details,
                },
            )
            if number % 25 == 0 or number == len(todo):
                log.info("[%d/%d] ok=%d failed=%d", number, len(todo), n_ok, n_fail)

    log.info("Done: %d labelled, %d failed -> %s", n_ok, n_fail, out_path)
    if n_fail:
        log.info("Failures logged in %s; re-run the same command to retry them.", fail_path)
    return LabelRunResult(n_ok, n_fail, out_path)
