# Step 00 — Scaffold

**Who:** 🤖 Claude · **Status:** done

## What was done
1. Created the folder layout (see the README table).
2. Wrote the **output schema** — `src/triage/schema.py`.
3. Wrote the **shared prompt** — `src/triage/prompts.py`.
4. Wrote stub scripts with TODOs pointing at the step that fills them.
5. Wrote the first ADRs in `docs/DECISIONS.md`.

## Why the schema came first
Every later component depends on one thing: *what does a correct answer look like?*

```
teacher labels ──► training data ──► fine-tuned model ──► gateway output
       ▲                                    ▲                    ▲
       └──────────── all validated against schema.py ────────────┘
```

If the schema changes later, the labelled data is invalid and must be regenerated (which costs
money). So it's designed up front, kept small, and **versioned** (`SCHEMA_VERSION`).

In your earlier GPT help-desk project, the prompt probably *described* the format in words.
Here the format is code (Pydantic), which gives us:
- automatic validation of every model output,
- a JSON Schema we can feed to constrained decoding (Step 10),
- exact field-by-field comparison for evals (Step 05).

## What you should learn here
- Read `schema.py` and `prompts.py` fully. Ask about anything unclear.
- Notice the schema is a **draft**: the category list will be finalised in Step 02 once we've
  seen real data. Designing labels without looking at data is a classic mistake.

## Decisions made
- [ADR-001](../DECISIONS.md#adr-001) — task & schema
- [ADR-002](../DECISIONS.md#adr-002) — base model size
- [ADR-003](../DECISIONS.md#adr-003) — where to train/serve
- [ADR-004](../DECISIONS.md#adr-004) — Python tooling
- [ADR-005](../DECISIONS.md#adr-005) — teacher model (**open — needs your input**)
- [ADR-006](../DECISIONS.md#adr-006) — one decision log file
