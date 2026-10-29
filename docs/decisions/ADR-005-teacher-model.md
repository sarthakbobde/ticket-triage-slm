# ADR-005 — Teacher model for labelling

**Status:** Proposed — **needs your decision before Step 03**

## Context
We'll label ~2–5k tickets with a large model; the small model learns to imitate it
(this is called **distillation**). The teacher's quality is the *ceiling* for the student.

Cost estimate: ~3k tickets × ~600 tokens in + ~300 out ≈ 1.8M in / 0.9M out tokens.
With a mid-tier API model that's typically **a few dollars**; with a frontier model, maybe $10–30.
(Check current prices before running.)

## Options
| Option | Quality | Cost | Notes |
|---|---|---|---|
| OpenAI GPT (you've used it) | high | low–mid | familiar API, structured outputs supported |
| Anthropic Claude | high | low–mid | strong at following schemas |
| Big open model via free/cheap API (e.g. Groq, OpenRouter, HF Inference) | medium–high | free–low | rate limits; fully "open-weight" story |
| Use dataset's existing labels only | n/a | free | no summary/reply/entities — not enough |

## ⚠️ Licence check
Some providers' terms restrict using outputs to train competing models. Read the terms of the
provider you pick. For a personal learning/portfolio project this is usually fine, but it's a
real question companies ask — note your answer here.

## Decision
_TBD — fill in after discussion._

## Consequences
_TBD_
