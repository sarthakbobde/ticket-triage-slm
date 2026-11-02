"""Label tickets with the teacher model -> data/labeled/<split>__<model>.jsonl (Step 03).

- The same SYSTEM_PROMPT as in training and serving (prompts.py).
- Every answer is checked with TicketTriage. If it is invalid we try once more, then log a failure.
- Safe to stop and start again: ids already in the output file are skipped, so a crash never costs
  money twice.
- Every row records the schema version, the teacher model and the token counts (for the cost report).
- Calls cost money: a remote server needs approved=True (the --yes option on the command line).
"""

import logging
import time

from pydantic import ValidationError

from triage import paths
from triage.errors import UserError
from triage.io import append_row, read_done_ids, read_jsonl
from triage.llm import make_client, require_approval_for_paid_calls, response_format_for
from triage.prompts import build_messages
from triage.schema import SCHEMA_VERSION, TicketTriage
from triage.settings import load_endpoint

log = logging.getLogger(__name__)

LABEL_SPLITS = ["val", "train", "train_extra", "train_extra2", "test"]


class LabelRunResult:
    """How a labelling run went."""

    def __init__(self, labelled, failed, out_path):
        self.labelled = labelled  # how many tickets got a label
        self.failed = failed  # how many tickets failed
        self.out_path = out_path  # the file the labels were written to


def label_one(client, endpoint, model, text):
    """Label one ticket. Returns (the label, details about the call).

    If the answer is not valid JSON we ask once more. If it is still invalid we raise an error.
    """
    messages = build_messages(text)
    response_format = response_format_for(endpoint.response_format)

    last_error = ""
    for attempt in range(2):
        started = time.perf_counter()

        # Some models (the "reasoning" ones) accept an extra setting. Only send it if the config has it.
        if endpoint.reasoning_effort:
            reply = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=endpoint.temperature,
                response_format=response_format,
                reasoning_effort=endpoint.reasoning_effort,
            )
        else:
            reply = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=endpoint.temperature,
                response_format=response_format,
            )

        # Write down how long it took and how many tokens were used.
        details = {}
        details["latency_s"] = round(time.perf_counter() - started, 2)
        details["input_tokens"] = None
        details["output_tokens"] = None
        if reply.usage:
            details["input_tokens"] = reply.usage.prompt_tokens
            details["output_tokens"] = reply.usage.completion_tokens

        raw = reply.choices[0].message.content
        if raw is None:
            raw = ""

        # Check the answer against our schema.
        try:
            label = TicketTriage.model_validate_json(raw)
            return label, details
        except ValidationError as error:
            first_problem = error.errors()[0]["msg"]
            last_error = f"{error.error_count()} errors: {first_problem} | raw={raw[:300]}"

    raise ValueError(last_error)


def list_models(client):
    """The model names this key is allowed to use, sorted."""
    names = []
    for model in client.models.list():
        names.append(model.id)
    names.sort()
    return names


def label_split(split, model=None, limit=None, approved=False, endpoint=None, client=None):
    """Label every ticket of one split (for example "train"). Pass a fake `client` in tests."""
    if split not in LABEL_SPLITS:
        raise UserError(f"Unknown split '{split}'. Choose from {LABEL_SPLITS}.")

    if endpoint is None:
        endpoint = load_endpoint("teacher")
    if not model:
        model = endpoint.model
    if not model:
        raise UserError("No model: pass --model or set teacher.model in configs/models.yaml.")

    tickets = read_jsonl(paths.PROCESSED / f"{split}.jsonl")
    if limit:
        tickets = tickets[:limit]  # [:limit] means "only the first `limit` tickets"

    safe_model = model.replace("/", "_")
    out_path = paths.LABELED / f"{split}__{safe_model}.jsonl"
    fail_path = paths.LABELED / f"failures__{safe_model}.jsonl"
    paths.LABELED.mkdir(parents=True, exist_ok=True)

    # Work out which tickets still need a label.
    done = read_done_ids(out_path)
    todo = []
    for ticket in tickets:
        if ticket["id"] not in done:
            todo.append(ticket)
    log.info("%s | %s: %d tickets, %d already labelled, %d to go", model, split, len(tickets), len(done), len(todo))
    if len(todo) == 0:
        return LabelRunResult(0, 0, out_path)

    # Money check: stop here if this would make paid calls that were not approved.
    require_approval_for_paid_calls(endpoint, len(todo), approved)
    if client is None:
        client = make_client(endpoint, max_retries=6, timeout=120)

    # If the config sets a limit of requests per minute, wait long enough between calls to stay under it.
    min_gap = 0.0
    if endpoint.rpm:
        min_gap = 60 / endpoint.rpm

    labelled = 0
    failed = 0
    number = 0
    last_call = 0.0
    with open(out_path, "a", encoding="utf-8") as out_file:
        with open(fail_path, "a", encoding="utf-8") as fail_file:
            for ticket in todo:
                number = number + 1

                # Wait if the previous call was too recent.
                wait = min_gap - (time.perf_counter() - last_call)
                if wait > 0:
                    time.sleep(wait)
                last_call = time.perf_counter()

                try:
                    label, details = label_one(client, endpoint, model, ticket["text"])
                except Exception as error:
                    # One bad ticket must not stop the whole run: write it to the failures file and go on.
                    failed = failed + 1
                    failure = {"id": ticket["id"], "split": split, "error": str(error)[:500]}
                    append_row(fail_file, failure)
                    log.warning("[%d/%d] %s FAILED: %s", number, len(todo), ticket["id"], str(error)[:120])
                    continue

                labelled = labelled + 1
                row = {}
                row["id"] = ticket["id"]
                row["schema_version"] = SCHEMA_VERSION
                row["teacher"] = model
                row["label"] = label.model_dump(mode="json")
                for key in details:
                    row[key] = details[key]
                append_row(out_file, row)

                if number % 25 == 0 or number == len(todo):
                    log.info("[%d/%d] ok=%d failed=%d", number, len(todo), labelled, failed)

    log.info("Done: %d labelled, %d failed -> %s", labelled, failed, out_path)
    if failed > 0:
        log.info("Failures logged in %s; run the same command again to retry them.", fail_path)
    return LabelRunResult(labelled, failed, out_path)
