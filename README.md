# Ticket Triage SLM

Fine-tune a **small open-weight model** (1.5B–3B) to triage customer-support tickets into
structured JSON, prove with evals that it is close to a large API model, and serve it cheaply
on a laptop GPU.

> The pitch this project is built to earn:
> *"My 3B model matches the big model on X% of tickets at 1/Nth the cost and lower latency."*

## Quick start

```bash
uv sync --extra serve        # install the package and its tools
uv run triage --help         # every pipeline step is a subcommand
uv run pytest                # tests (no network, no GPU, no paid API)
uv run ruff check src tests  # lint
```

The pipeline, in order (details and reasoning in [docs/steps/](docs/steps/)):

| Step | Command |
|---|---|
| Download and profile the dataset | `triage data download` |
| Build test / val / train splits | `triage data splits` |
| Label with the teacher model (costs money, needs `--yes`) | `triage label --split train --yes` |
| Find extra rare-class tickets | `triage data mine --round 2` |
| Merge `hr` into `customer_service` (ADR-013) | `triage data merge-hr` |
| Build the hand-labelling page for the gold set | `triage data gold-sheet` |
| Build the Kaggle training files | `triage data training-data` |
| Collect a model's answers on the gold set | `triage eval run --config base_qwen --run NAME` |
| Score those answers (free) | `triage eval score --run NAME` |

## How the repo is organised

| Path | What lives there |
|---|---|
| `src/triage/` | The Python package. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the layout and conventions. |
| `tests/` | Unit tests. They run on tiny fake data and a fake model client. |
| `configs/` | `models.yaml`: every model is an OpenAI-compatible endpoint, switched by config, not code. |
| `notebooks/` | Exploration and the Kaggle training notebook. Notebooks explore; the package is the pipeline. |
| `evals/` | Output folder only: `runs/` (raw model answers, git-ignored) and `reports/` (scored reports). |
| `data/` | Datasets and labels. **Git-ignored**, except the reviewed gold labels. |
| `docs/steps/` | One guide per step: what to do, how, and what you learn. |
| `docs/DECISIONS.md` | Decision log (ADRs): why each choice was made and what was rejected. |
| `docs/STEPS.md` | The roadmap and progress. |
| `serve/` | FastAPI gateway in front of a local llama.cpp / Ollama server (Step 11). |

## Hardware this was designed for

- Local: NVIDIA GTX 1650 (4 GB), 16 GB RAM → **inference only** (quantized GGUF).
- Training: Kaggle Notebooks (2× T4, 16 GB each, ~30 free GPU-hours/week).
- See [ADR-003](docs/DECISIONS.md#adr-003).
