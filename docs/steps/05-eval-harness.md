# Step 05: Eval harness

**Who:** 🤝 · **Status:** done · **Cost:** $0 to build (running the teacher through it ≈ $0.40)

## The idea
Before any training we need a ruler. Every model (the untrained base, the teacher, the fine-tune,
each GGUF quantisation) is run on the **same 200 gold tickets** and scored with the **same code**,
so the numbers can be compared fairly.

```
gold tickets ─► evals/run_model.py ─► evals/runs/<run>.jsonl   (raw answers, not parsed)
                                              │
 data/gold/gold_labels.jsonl ──────────────► evals/score.py ─► evals/reports/<run>.md + .json
```

## Use it
```bash
# 1. collect answers (local models are free; a remote endpoint needs --yes because it costs money)
uv run python evals/run_model.py --config student_local --run base-untrained --format none
uv run python evals/run_model.py --config teacher --run teacher-gpt-4.1 --format json_schema --yes
# 2. score (free)
uv run python evals/score.py --run teacher-gpt-4.1
# 3. check the metric code itself
uv run pytest
```

## What is measured (src/triage/metrics.py)
| Metric | Why |
|---|---|
| Valid JSON, fits the schema | A model that cannot produce the contract is useless, however smart. |
| Category accuracy and **macro-F1**, per-class table, confusion matrix | Macro-F1 treats every class equally, so small classes (billing 8, outage 13) cannot hide behind `technical`. |
| Priority accuracy and **off-by-one** rate | low vs high is a worse mistake than low vs medium. |
| Sentiment accuracy | Only 2 positive gold tickets, so read the per-class table, not just the average. |
| Products F1 | Set overlap per ticket; both empty counts as correct. |
| Reply promises refund/timeline | Rough keyword check of a prompt rule, not a judge. |
| Latency p50 / p95 | Feeds the cost report in Step 12. |

**Rule:** an invalid answer counts as wrong for every field. Otherwise a model could look good
by only answering the easy tickets.

## Choices worth knowing
- Raw text is saved and parsed at scoring time, so a better parser never needs new model calls.
- `--format none` is the honest test for a base model; `json_schema` shows what constrained
  decoding adds (compare the two in Step 06).
- `tests/test_metrics.py` checks every metric against scikit-learn, and caught a real bug
  (a percentile rounding error) while this step was built.
- The gold set is small (200). Differences of a few points between models are noise; billing
  and outage scores can move 8–12 points on a single ticket.
- `summary` quality is not scored yet: it needs a judge model or human review (a later option).

## Findings
1. Smoke test with fake answers: a perfect run scores 100% everywhere; a run with 10 broken
   JSON answers and 6 schema breaks reports exactly 95% JSON-valid and 92% schema-valid.
2. Next: Step 06 runs the untrained base model through this harness.
