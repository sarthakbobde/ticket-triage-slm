# Roadmap

Legend: 🧑 = you do it · 🤖 = Claude does it (after explaining) · 🤝 = together

Each step has a doc in `docs/steps/`. Tick the box when done.

## Phase 0 — Foundations
- [x] 🤖 **00 Scaffold** — folder structure, schema, prompt, docs. → [00-scaffold.md](steps/00-scaffold.md)
- [x] 🧑 **01 Environment setup** — git, uv, venv, HF cache on E:. → [01-environment-setup.md](steps/01-environment-setup.md)

## Phase 1 — Data
- [x] 🤝 **02 Pick & explore a dataset** — find public support tickets, look at them, finalise the label set. → [02-dataset.md](steps/02-dataset.md)
- [x] 🤝 **03 Split & label with a teacher model** — build the splits (ADR-012), pilot two teachers, label ~3.3k tickets (distillation). → [03-teacher-labels.md](steps/03-teacher-labels.md)
- [x] 🧑 **04 Audit labels & gold set** — gold set labelled (drafted by Claude, reviewed by you); `data/labeled/merged/` is the training input. *Teacher-vs-gold check deferred: run `triage eval run --config teacher --run teacher-gpt-4.1 --yes` (≈ $0.40) if training labels look doubtful.*

## Phase 2 — Baseline & evals (before any training!)
- [x] 🤝 **05 Eval harness** — metrics: JSON validity, per-field accuracy, macro-F1, latency. → [05-eval-harness.md](steps/05-eval-harness.md)
- [x] 🧑 **06 Baseline** — run two *untrained* small models through the harness; Qwen2.5-3B chosen (ADR-014). *Teacher row deferred.* → [06-baseline.md](steps/06-baseline.md)

## Phase 3 — Fine-tuning
- [ ] 🤝 **07 QLoRA on Kaggle** — train, track loss, save the adapter. *Data script and notebook written; waiting for the Kaggle run.* → [07-qlora-kaggle.md](steps/07-qlora-kaggle.md)
- [ ] 🧑 **08 Evaluate the fine-tune** — compare against baseline. Iterate on data, not just hyperparams.

## Phase 4 — Inference
- [ ] 🤝 **09 Merge + quantize to GGUF** — Q4/Q5/Q8, measure quality loss per level.
- [ ] 🧑 **10 Serve locally** — llama.cpp / Ollama on the GTX 1650.
- [ ] 🤝 **11 Gateway** — FastAPI, schema validation, fallback to teacher when low-confidence, logging.

## Phase 5 — Show it off
- [ ] 🤝 **12 Cost & latency report** — $/1k tickets vs API, p50/p95 latency, quality table.
- [ ] 🧑 **13 Write-up** — README results section, demo GIF, what you'd do next.
