# ADR-001 — Task and output schema

**Status:** Accepted (category list is draft until Step 02)

## Context
We need a task that (a) product companies really pay API costs for, (b) has objectively
checkable outputs so we can measure quality, and (c) is narrow enough that a small model can
learn it well. You've built a help-desk project with GPT before, so you know the domain.

## Options
1. **Free-text reply generation only** — easy to build, but hard to evaluate (needs human or
   LLM judges for everything). Small models are weakest here.
2. **Pure classification** — easy to evaluate, but too simple; a fine-tuned BERT would do it and
   it wouldn't teach you anything about LLMs.
3. **Structured JSON triage (classification + extraction + short generation)** — mostly
   exact-match evaluable, with one generative field for realism.

## Decision
Option 3. One ticket in → one JSON object out, defined in `src/triage/schema.py`:

| Field | Type | How it's evaluated |
|---|---|---|
| `category` | enum | accuracy, macro-F1 |
| `priority` | enum (low/medium/high/urgent) | accuracy + "off by one level" rate |
| `sentiment` | enum | accuracy |
| `entities` | object (order id, product, ...) | per-field exact match |
| `summary` | short string | LLM-judge / ROUGE (secondary) |
| `suggested_reply` | string | LLM-judge (secondary) |

The schema carries `SCHEMA_VERSION`. Any change bumps it and means re-labelling.

## Consequences
- ✅ ~80% of the output is exactly measurable — cheap, trustworthy evals.
- ✅ The same Pydantic model gives validation, JSON Schema for constrained decoding, and eval diffing.
- ⚠️ Category list must be finalised after looking at data (Step 02), not before.
- ⚠️ `suggested_reply` makes outputs long → slower inference and training. If latency targets are
  missed, we may drop it from the small model and keep it as a fallback-only feature.
