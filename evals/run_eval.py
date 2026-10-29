"""Steps 05/06/08 — score any OpenAI-compatible model on data/processed/test.jsonl

TODO (Step 05):
- args: --model-config <name from configs/models.yaml> --limit N
- metrics: JSON-valid rate, schema-valid rate, per-field accuracy, category macro-F1,
  priority off-by-one rate, entity exact match, p50/p95 latency, tokens/sec
- save predictions + metrics to evals/results/<model>-<timestamp>/
"""
