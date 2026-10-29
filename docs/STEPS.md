# Roadmap

Legend: 🧑 = you do it · 🤖 = Claude does it (after explaining) · 🤝 = together

Each step has a doc in `docs/steps/`. Tick the box when done.

## Phase 0 — Foundations
- [x] 🤖 **00 Scaffold** — folder structure, schema, prompt, docs. → [00-scaffold.md](steps/00-scaffold.md)
- [x] 🧑 **01 Environment setup** — git, uv, venv, HF cache on E:. → [01-environment-setup.md](steps/01-environment-setup.md)

## Phase 1 — Data
- [ ] 🤝 **02 Pick & explore a dataset** — find public support tickets, look at them, finalise the label set. → [02-dataset.md](steps/02-dataset.md)
- [ ] 🤝 **03 Label with a teacher model** — a large model produces gold JSON (distillation).
- [ ] 🧑 **04 Clean & split** — validate JSON, dedupe, train/val/test split. Test set is frozen forever.

## Phase 2 — Baseline & evals (before any training!)
- [ ] 🤝 **05 Eval harness** — metrics: JSON validity, per-field accuracy, macro-F1, latency.
- [ ] 🧑 **06 Baseline** — run the *untrained* small model + the teacher through the harness.

## Phase 3 — Fine-tuning
- [ ] 🤝 **07 QLoRA on Kaggle** — train, track loss, save the adapter.
- [ ] 🧑 **08 Evaluate the fine-tune** — compare against baseline. Iterate on data, not just hyperparams.

## Phase 4 — Inference
- [ ] 🤝 **09 Merge + quantize to GGUF** — Q4/Q5/Q8, measure quality loss per level.
- [ ] 🧑 **10 Serve locally** — llama.cpp / Ollama on the GTX 1650.
- [ ] 🤝 **11 Gateway** — FastAPI, schema validation, fallback to teacher when low-confidence, logging.

## Phase 5 — Show it off
- [ ] 🤝 **12 Cost & latency report** — $/1k tickets vs API, p50/p95 latency, quality table.
- [ ] 🧑 **13 Write-up** — README results section, demo GIF, what you'd do next.
