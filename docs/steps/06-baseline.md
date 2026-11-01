# Step 06: Baseline (untrained models)

**Who:** 🧑/🤝 · **Status:** done · **Cost:** $0 (everything ran locally)

## The idea
Measure the untrained candidates **before** any training. Without this "before" number we could
not tell whether fine-tuning helped, or by how much. It also decides which base model we train.

## What was set up
- **Ollama 0.40.13** in `E:\Ollama`, models in `E:\Ollama\models` (set with the user environment
  variable `OLLAMA_MODELS`). Models: `qwen2.5:3b-instruct` (1.9 GB), `llama3.2:3b` (2.0 GB).
- `configs/models.yaml` sections `base_qwen` and `base_llama` point at `http://localhost:11434/v1`.
- Four runs on the 200 gold tickets (plain prompt vs schema-enforced, per model):
  ```bash
  uv run triage eval run --config base_qwen  --run base-qwen-none    --format none
  uv run triage eval run --config base_llama --run base-llama-none   --format none
  uv run triage eval run --config base_qwen  --run base-qwen-schema  --format json_schema
  uv run triage eval run --config base_llama --run base-llama-schema --format json_schema
  uv run triage eval score --run <each run name>
  ```
  Reports are in `evals/reports/`; the raw answers (`evals/runs/`) are git-ignored.

## Result
See [ADR-014](../DECISIONS.md#adr-014) for the full table and the decision: **Qwen2.5-3B-Instruct**.
Qwen: valid JSON 100% even untrained, category 75.5%, priority 40.5%. Llama: priority 76% but needs
schema enforcement, is bigger and spills out of the 4 GB card.

## Things that went wrong (and what they taught)
1. After a resume, 90 Llama rows were saved as errors ("Connection error", "model not found")
   because the Ollama tray app had restarted its own server **without** our model folder setting.
   The failed rows were counted as "done". Fixed: failed calls are retried on resume, and the
   scorer keeps the latest row per ticket. Lesson: a harness must never treat a failure as a result.
2. Llama's plain-prompt validity was 2%. We checked the raw text before believing it
   (missing final `}`), and kept the strict rule instead of quietly repairing the output.

## Findings
1. Winner and why: Qwen2.5-3B, because it fits the GPU and already outputs valid JSON.
2. Numbers to beat in Step 08: category 75.5% / macro-F1 62.5%, priority 40.5%, sentiment 63.0%, products F1 65.0%.
3. Biggest untrained gaps: `customer_service` recall (about 3%), `outage` recall (46%), priority.
4. Licence to read before publishing anything: `qwen-research` (see ADR-014).
