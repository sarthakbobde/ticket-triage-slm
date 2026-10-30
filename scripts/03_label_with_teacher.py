"""Step 03b: label tickets with the teacher model -> data/labeled/<split>__<model>.jsonl

Run (see docs/steps/03-teacher-labels.md for the full order):
  uv run python scripts/03_label_with_teacher.py --list-models
  uv run python scripts/03_label_with_teacher.py --split val --limit 50 --model <candidate>
  uv run python scripts/03_label_with_teacher.py --split train

- Same SYSTEM_PROMPT as training and serving (src/triage/prompts.py).
- Every answer is validated with TicketTriage; invalid -> one retry -> failures file.
- Resumable: ids already in the output file are skipped, so a crash never costs money twice.
- Every row records SCHEMA_VERSION, the teacher model and token usage (for the cost report).
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
from pydantic import ValidationError

from triage.prompts import build_messages
from triage.schema import SCHEMA_VERSION, TicketTriage

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "labeled"


def load_teacher_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["teacher"]


def make_client(cfg: dict) -> OpenAI:
    load_dotenv(ROOT / ".env")
    key = os.environ.get(cfg["api_key_env"])
    if not key:
        sys.exit(f"Set {cfg['api_key_env']} in .env first (see .env.example).")
    # max_retries: the SDK backs off and retries on 429 (rate limit) and 5xx by itself.
    return OpenAI(base_url=cfg["base_url"], api_key=key, max_retries=6, timeout=120)


def response_format(cfg: dict) -> dict:
    """Constrained decoding: the API itself is told the JSON shape, so most outputs are valid."""
    if cfg.get("response_format") == "json_object":
        return {"type": "json_object"}  # fallback if the provider rejects our JSON Schema
    return {
        "type": "json_schema",
        "json_schema": {"name": "TicketTriage", "schema": TicketTriage.model_json_schema()},
    }


def label_one(client: OpenAI, cfg: dict, model: str, text: str) -> tuple[TicketTriage, dict]:
    """One ticket -> validated label. Retries once on invalid JSON, then raises."""
    extra = {"reasoning_effort": cfg["reasoning_effort"]} if cfg.get("reasoning_effort") else {}
    last_error = None
    for _ in range(2):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=build_messages(text),
            temperature=cfg.get("temperature", 0),
            response_format=response_format(cfg),
            **extra,
        )
        meta = {
            "latency_s": round(time.perf_counter() - t0, 2),
            "input_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "output_tokens": resp.usage.completion_tokens if resp.usage else None,
        }
        raw = resp.choices[0].message.content or ""
        try:
            return TicketTriage.model_validate_json(raw), meta
        except ValidationError as e:
            last_error = f"{e.error_count()} errors: {e.errors()[0]['msg']} | raw={raw[:300]}"
    raise ValueError(last_error)


def read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.open(encoding="utf-8")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "train", "test"])
    ap.add_argument("--model", help="override teacher.model in configs/models.yaml (pilot)")
    ap.add_argument("--limit", type=int, help="only the first N tickets of the split (pilot)")
    ap.add_argument("--list-models", action="store_true", help="print model ids the key can use")
    args = ap.parse_args()

    cfg = load_teacher_config()
    client = make_client(cfg)

    if args.list_models:
        for m in sorted(m.id for m in client.models.list()):
            print(m)
        return
    if not args.split:
        ap.error("--split is required")

    model = args.model or cfg["model"]
    if not model:
        sys.exit("No model: pass --model or set teacher.model in configs/models.yaml.")

    tickets = [json.loads(line) for line in (SPLITS / f"{args.split}.jsonl").open(encoding="utf-8")]
    if args.limit:
        tickets = tickets[: args.limit]

    safe_model = model.replace("/", "_")
    out_path = OUT_DIR / f"{args.split}__{safe_model}.jsonl"
    fail_path = OUT_DIR / f"failures__{safe_model}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = read_ids(out_path)
    todo = [t for t in tickets if t["id"] not in done]
    print(f"{model} | {args.split}: {len(tickets)} tickets, {len(done)} already labelled, {len(todo)} to go")

    min_gap = 60 / cfg["rpm"] if cfg.get("rpm") else 0  # stay under the free-tier rate limit
    n_ok = n_fail = 0
    last_call = 0.0
    with out_path.open("a", encoding="utf-8") as out, fail_path.open("a", encoding="utf-8") as fail:
        for i, t in enumerate(todo, 1):
            time.sleep(max(0.0, min_gap - (time.perf_counter() - last_call)))
            last_call = time.perf_counter()
            try:
                label, meta = label_one(client, cfg, model, t["text"])
            except Exception as e:  # noqa: BLE001 — log and move on; never lose the whole run
                n_fail += 1
                fail.write(json.dumps({"id": t["id"], "split": args.split, "error": str(e)[:500]}) + "\n")
                fail.flush()
                print(f"[{i}/{len(todo)}] {t['id']} FAILED: {str(e)[:120]}")
                continue
            n_ok += 1
            row = {
                "id": t["id"],
                "schema_version": SCHEMA_VERSION,
                "teacher": model,
                "label": label.model_dump(mode="json"),
                **meta,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()  # written line by line, so a crash loses at most one ticket
            if i % 25 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] ok={n_ok} failed={n_fail}")

    print(f"Done: {n_ok} labelled, {n_fail} failed -> {out_path.relative_to(ROOT)}")
    if n_fail:
        print(f"Failures logged in {fail_path.relative_to(ROOT)}; re-run the same command to retry them.")


if __name__ == "__main__":
    main()
