# ADR-004 — Python tooling

**Status:** Accepted

## Decisions
### uv instead of pip/conda
- Fast installs, a **lockfile** (`uv.lock`) for exact reproducibility, one tool for venv + deps.
- Rejected conda: heavy, and we don't need its CUDA management locally (training is on Kaggle).
- Rejected plain pip + requirements.txt: no lockfile by default → "works on my machine" drift.

### Dependency groups
- **Core** (local): pydantic, httpx, openai, datasets, pandas, pyyaml, python-dotenv, tqdm.
- **serve** (local): fastapi, uvicorn.
- **Training deps are NOT installed locally** (torch, transformers, peft, trl, bitsandbytes).
  They're installed inside the Kaggle notebook. Installing CUDA torch locally would cost ~3 GB
  and we'd never use it for training.

### Pydantic for the schema
Gives validation + JSON Schema + typed Python objects from one definition. See ADR-001.

### OpenAI-compatible API everywhere
The teacher (whichever provider), Ollama, llama.cpp's `llama-server`, and vLLM all expose an
OpenAI-compatible `/v1/chat/completions` endpoint. So the eval harness and the gateway use **one
client** and switch models by changing `base_url` + `model` in config.
- ✅ Swapping teacher/student/backend is a config change, not a code change.
- ✅ Mirrors how companies abstract model providers.

### `src/` layout
Code in `src/triage/` is importable from scripts, evals and the server alike (`from triage.schema import ...`).
