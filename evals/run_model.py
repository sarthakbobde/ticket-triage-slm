"""Step 05a: ask a model to triage the 200 gold tickets and save its RAW answers.

Run (each model gets its own run name, so runs never overwrite each other):
  uv run python evals/run_model.py --config student_local --run base-qwen-untrained
  uv run python evals/run_model.py --config teacher --run teacher-gpt-4.1 --yes

Then score it (free, no model calls):
  uv run python evals/score.py --run base-qwen-untrained

- Models come from configs/models.yaml (every model is an OpenAI-compatible endpoint, ADR-004).
- We save the raw text, not a parsed answer. Scoring decides if it is valid, so a model that
  breaks the JSON cannot hide it.
- --format decides how hard we push the model towards valid JSON:
    none        just the prompt (honest test for an untrained base model)
    json_object the API promises *some* JSON
    json_schema the API enforces our exact schema (constrained decoding)
- Resumable: ids already in the run file are skipped.
- Remote endpoints (anything not on localhost) cost money, so they need --yes.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from triage.prompts import build_messages
from triage.schema import TicketTriage

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "processed" / "test.jsonl"
RUNS_DIR = ROOT / "evals" / "runs"


def load_config(name):
    configs = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    if name not in configs:
        sys.exit(f"No '{name}' section in configs/models.yaml. Found: {list(configs)}")
    return configs[name]


def make_client(config):
    load_dotenv(ROOT / ".env")
    key = "not-needed"  # local servers ignore the key, but the client wants some text
    if config.get("api_key_env"):
        key = os.environ.get(config["api_key_env"], "")
        if not key:
            sys.exit(f"Set {config['api_key_env']} in .env first.")
    return OpenAI(base_url=config["base_url"], api_key=key, max_retries=3, timeout=300)


def is_local(base_url):
    return "localhost" in base_url or "127.0.0.1" in base_url


def response_format_for(kind):
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "TicketTriage", "schema": TicketTriage.model_json_schema()},
        }
    return None


def load_gold_tickets():
    tickets = []
    with open(TEST_FILE, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row["gold"]:
                tickets.append({"id": row["id"], "text": row["text"]})
    return tickets


def read_done_ids(path):
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as file:
            for line in file:
                done.add(json.loads(line)["id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="section of configs/models.yaml, e.g. teacher")
    parser.add_argument("--run", required=True, help="name for this run, e.g. teacher-gpt-4.1")
    parser.add_argument("--model", help="override the model name in the config")
    parser.add_argument("--format", choices=["none", "json_object", "json_schema"], default="none")
    parser.add_argument("--limit", type=int, help="only the first N gold tickets")
    parser.add_argument("--yes", action="store_true", help="allow calls to a remote (paid) endpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    model = args.model or config["model"]
    tickets = load_gold_tickets()
    if args.limit:
        tickets = tickets[: args.limit]

    out_path = RUNS_DIR / f"{args.run}.jsonl"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    done = read_done_ids(out_path)
    todo = []
    for ticket in tickets:
        if ticket["id"] not in done:
            todo.append(ticket)
    print(f"{model} | {len(tickets)} gold tickets, {len(done)} done, {len(todo)} to go")
    if len(todo) == 0:
        return

    if not is_local(config["base_url"]) and not args.yes:
        sys.exit(f"{config['base_url']} is a remote endpoint: {len(todo)} paid calls. Re-run with --yes.")

    client = make_client(config)
    response_format = response_format_for(args.format)
    min_gap = 0.0
    if config.get("rpm"):
        min_gap = 60 / config["rpm"]

    last_call = 0.0
    with open(out_path, "a", encoding="utf-8") as out:
        for number, ticket in enumerate(todo, 1):
            wait = min_gap - (time.perf_counter() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.perf_counter()

            row = {"id": ticket["id"], "model": model, "format": args.format}
            started = time.perf_counter()
            try:
                request = {
                    "model": model,
                    "messages": build_messages(ticket["text"]),
                    "temperature": 0,
                }
                if response_format is not None:
                    request["response_format"] = response_format
                reply = client.chat.completions.create(**request)
                row["raw"] = reply.choices[0].message.content or ""
                if reply.usage:
                    row["input_tokens"] = reply.usage.prompt_tokens
                    row["output_tokens"] = reply.usage.completion_tokens
            except Exception as error:  # keep going: a failed call is scored as a failure
                row["raw"] = None
                row["error"] = str(error)[:300]
            row["latency_s"] = round(time.perf_counter() - started, 3)

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if number % 20 == 0 or number == len(todo):
                print(f"[{number}/{len(todo)}]")

    print(f"Saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
