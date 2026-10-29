# Ticket Triage SLM

Fine-tune a **small open-weight model** (1.5B–3B) to triage customer-support tickets into
structured JSON, prove with evals that it is close to a large API model, and serve it cheaply
on a laptop GPU.

> The pitch this project is built to earn:
> *"My 3B model matches the big model on X% of tickets at 1/Nth the cost and lower latency."*

## How this repo is organised for learning

| Folder | What lives there |
|---|---|
| `docs/steps/` | One file per step: **what** to do, **how**, and **what you learn**. Work through them in order. |
| `docs/decisions/` | ADRs (Architecture Decision Records): **why** each choice was made and what was rejected. |
| `src/triage/` | Shared library code: the output schema and prompts (the "contract"). |
| `scripts/` | Numbered pipeline scripts: download → label → split. |
| `notebooks/` | Training notebook that runs on Kaggle (free T4 GPUs). |
| `evals/` | Evaluation harness: compare base vs fine-tuned vs teacher. |
| `serve/` | FastAPI gateway in front of a local llama.cpp / Ollama server. |
| `configs/` | YAML configs (training hyperparameters, model names). |
| `data/` | Datasets — **git-ignored**, never committed. |

## Roadmap

See [docs/STEPS.md](docs/STEPS.md) for the full checklist and [docs/decisions/](docs/decisions/) for the reasoning.

## Hardware this was designed for

- Local: NVIDIA GTX 1650 (4 GB), 16 GB RAM → **inference only** (quantized GGUF).
- Training: Kaggle Notebooks (2× T4, 16 GB each, ~30 free GPU-hours/week).
- See [ADR-003](docs/decisions/ADR-003-compute-train-in-cloud-serve-locally.md).
