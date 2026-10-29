# Decision Log

Every project decision lives in this one file. Each entry is an **ADR** (Architecture Decision
Record): the context, the options considered, what was chosen, and the consequences.

**Rules**
- Never delete or rewrite an accepted decision. If it changes, add a new ADR and mark the old one
  `Superseded by ADR-XXX`. The history is the point.
- Add new decisions at the bottom with the next number, and add a row to the index.
- Link to a decision as `docs/DECISIONS.md#adr-003`.

## Index

| ADR | Decision | Status | Date |
|---|---|---|---|
| [001](#adr-001) | Task = ticket triage; output = versioned Pydantic schema | Accepted (categories draft until Step 02) | 2026-10-29 |
| [002](#adr-002) | Base model 1.5B–3B; family picked at Step 06 | Accepted | 2026-10-29 |
| [003](#adr-003) | Train on Kaggle, serve on laptop via GGUF | Accepted | 2026-10-29 |
| [004](#adr-004) | uv, Pydantic, OpenAI-compatible clients everywhere | Accepted | 2026-10-29 |
| [005](#adr-005) | Which large model labels the data | **Proposed — needs you** | — |
| [006](#adr-006) | All decisions kept in one file | Accepted | 2026-10-29 |

## Template

```
<a id="adr-NNN"></a>
## ADR-NNN — Title
**Status:** Proposed | Accepted | Superseded by ADR-XXX · **Date:** YYYY-MM-DD · **Step:** NN

### Context
What problem or constraint forced a decision.
### Options
What was considered, with pros and cons.
### Decision
What we chose.
### Consequences
What this makes easier or harder.
```

---

<a id="adr-001"></a>
## ADR-001 — Task and output schema
**Status:** Accepted (category list is draft until Step 02) · **Date:** 2026-10-29 · **Step:** 00

### Context
We need a task that (a) product companies really pay API costs for, (b) has objectively
checkable outputs so we can measure quality, and (c) is narrow enough that a small model can
learn it well. You've built a help-desk project with GPT before, so you know the domain.

### Options
1. **Free-text reply generation only.** Easy to build, but hard to evaluate (needs human or
   LLM judges for everything). Small models are weakest here.
2. **Pure classification.** Easy to evaluate, but too simple: a fine-tuned BERT would do it and
   it wouldn't teach you anything about LLMs.
3. **Structured JSON triage (classification + extraction + short generation).** Mostly
   exact-match evaluable, with one generative field for realism.

### Decision
Option 3. One ticket in, one JSON object out, defined in `src/triage/schema.py`:

| Field | Type | How it's evaluated |
|---|---|---|
| `category` | enum | accuracy, macro-F1 |
| `priority` | enum (low/medium/high/urgent) | accuracy + "off by one level" rate |
| `sentiment` | enum | accuracy |
| `entities` | object (order id, product, ...) | per-field exact match |
| `summary` | short string | LLM-judge / ROUGE (secondary) |
| `suggested_reply` | string | LLM-judge (secondary) |

The schema carries `SCHEMA_VERSION`. Any change bumps it and means re-labelling.

### Consequences
- ✅ ~80% of the output is exactly measurable, so evals are cheap and trustworthy.
- ✅ The same Pydantic model gives validation, JSON Schema for constrained decoding, and eval diffing.
- ⚠️ The category list must be finalised after looking at data (Step 02), not before.
- ⚠️ `suggested_reply` makes outputs long, so inference and training are slower. If latency
  targets are missed, we may drop it from the small model and keep it as a fallback-only feature.

---

<a id="adr-002"></a>
## ADR-002 — Base model size
**Status:** Accepted (size). Exact model family chosen with data in Step 06. · **Date:** 2026-10-29 · **Step:** 00

### Context
- Local GPU: GTX 1650, **4 GB VRAM**, no tensor cores, no bf16.
- Training GPU: Kaggle T4, 16 GB.
- The product goal is "cheap and fast"; a big model defeats the purpose.

Rough memory maths (worth memorising):
- Weights ≈ parameters × bytes per parameter.
  - 3B in FP16 = 3 × 2 = **6 GB**, which doesn't fit locally.
  - 3B in 4-bit ≈ 3 × 0.5 ≈ **1.8–2 GB** (+ overhead), which fits in 4 GB with room for context.
  - 7B in 4-bit ≈ **4–4.5 GB**, which doesn't fully fit; needs CPU offload, so it's slow.
- QLoRA training ≈ 4-bit weights + small LoRA adapters + activations, so 3B trains comfortably on a T4.

### Options
| Size | Fits 1650 (Q4)? | Trains on T4? | Expected quality on narrow task |
|---|---|---|---|
| 0.5B–1B | easily | fast | ok for classification, weak replies |
| **1.5B–3B** | **yes** | **yes** | **good after fine-tuning** |
| 7B–8B | partial offload | tight | best, but slow locally |

### Decision
Target **1.5B–3B instruct models**. Candidate families (verify on Hugging Face before choosing):
Qwen (1.5B/3B), Llama 3.2 3B, Gemma (2B-class), SmolLM. We'll baseline 2 candidates untrained
in Step 06 and fine-tune the better one.

### Consequences
- ✅ The whole loop (train → quantize → serve) works on free hardware.
- ✅ Makes the cost story dramatic compared with a frontier API model.
- ⚠️ `suggested_reply` quality will lag the teacher. That's expected, and it's why the gateway has a fallback.
- ⚠️ Licences differ by family (Llama/Gemma have custom licences; Qwen is mostly Apache-2.0). Check before choosing.

---

<a id="adr-003"></a>
## ADR-003 — Train in the cloud, serve locally
**Status:** Accepted · **Date:** 2026-10-29 · **Step:** 00

### Context
Local machine: GTX 1650 4 GB, Ryzen 5 5600H, 16 GB RAM, C: nearly full, E: 174 GB free.
Budget: free.

### Options
| Option | Train | Serve | Cost |
|---|---|---|---|
| All local | ❌ 4 GB too small, no bf16 | ✅ | free |
| **Kaggle train + local serve** | ✅ T4×2, 30 h/week | ✅ GGUF on 1650 | **free** |
| Colab train + local serve | ✅ T4, session limits | ✅ | free (less reliable) |
| Rented GPU (RunPod etc.) | ✅ fastest | ✅ | ~$0.2–0.5/h |

### Decision
- **Train on Kaggle** (Colab as backup).
- **Serve locally** with llama.cpp / Ollama using a quantized GGUF.
- Optionally run **vLLM on Kaggle** for a throughput benchmark. This contrasts "edge/laptop
  deployment" with "server deployment", which is a good talking point.

### Consequences
- ✅ Zero cost.
- ✅ You learn the real-world handoff: train in one environment, export, deploy in another.
- ⚠️ Training code must be a notebook that runs top-to-bottom on Kaggle with no local files
  except the uploaded dataset. Keep data in a Kaggle Dataset or on the HF Hub.
- ⚠️ Kaggle sessions time out (~12 h). Save checkpoints to `/kaggle/working` and push adapters to the HF Hub.

---

<a id="adr-004"></a>
## ADR-004 — Python tooling
**Status:** Accepted · **Date:** 2026-10-29 · **Step:** 00

### Context
We need a reproducible local environment, one schema definition shared by every component, and
a way to call several different models (teacher, base candidates, fine-tune) without rewriting code.

### Options and decisions
**uv instead of pip/conda**
- Fast installs, a **lockfile** (`uv.lock`) for exact reproducibility, one tool for venv + deps.
- Rejected conda: heavy, and we don't need its CUDA management locally (training is on Kaggle).
- Rejected plain pip + requirements.txt: no lockfile by default, so "works on my machine" drift.

**Dependency groups**
- **Core** (local): pydantic, httpx, openai, datasets, pandas, pyyaml, python-dotenv, tqdm.
- **serve** (local): fastapi, uvicorn.
- **Training deps are NOT installed locally** (torch, transformers, peft, trl, bitsandbytes).
  They're installed inside the Kaggle notebook. CUDA torch locally would cost ~3 GB and never be
  used for training.

**Pydantic for the schema**
Validation + JSON Schema + typed Python objects from one definition (see ADR-001).

**OpenAI-compatible API everywhere**
The teacher (whichever provider), Ollama, llama.cpp's `llama-server`, and vLLM all expose an
OpenAI-compatible `/v1/chat/completions` endpoint. So the eval harness and the gateway use
**one client** and switch models by changing `base_url` + `model` in `configs/models.yaml`.

**`src/` layout**
Code in `src/triage/` is importable from scripts, evals and the server alike
(`from triage.schema import ...`).

### Consequences
- ✅ Swapping teacher/student/backend is a config change, not a code change. This mirrors how
  companies abstract model providers.
- ✅ Anyone can recreate the exact environment with `uv sync`.
- ⚠️ The local environment can't run training code; that only works on Kaggle.

---

<a id="adr-005"></a>
## ADR-005 — Teacher model for labelling
**Status:** Proposed — **needs your decision before Step 03** · **Date:** — · **Step:** 03

### Context
We'll label ~2–5k tickets with a large model; the small model learns to imitate it
(this is called **distillation**). The teacher's quality is the *ceiling* for the student.

Cost estimate: ~3k tickets × ~600 tokens in + ~300 out ≈ 1.8M in / 0.9M out tokens.
With a mid-tier API model that's typically **a few dollars**; with a frontier model, maybe
$10–30. Check current prices before running.

### Options
| Option | Quality | Cost | Notes |
|---|---|---|---|
| OpenAI GPT (you've used it) | high | low–mid | familiar API, structured outputs supported |
| Anthropic Claude | high | low–mid | strong at following schemas |
| Big open model via free/cheap API (e.g. Groq, OpenRouter, HF Inference) | medium–high | free–low | rate limits; fully "open-weight" story |
| Use dataset's existing labels only | n/a | free | no summary/reply/entities, so not enough |

**⚠️ Licence check:** some providers' terms restrict using outputs to train competing models.
Read the terms of the provider you pick. For a personal learning/portfolio project this is
usually fine, but it's a real question companies ask, so note your answer here.

### Decision
_TBD. Fill in after discussion._

### Consequences
_TBD_

---

<a id="adr-006"></a>
## ADR-006 — Keep all decisions in one file
**Status:** Accepted · **Date:** 2026-10-29 · **Step:** 00

### Context
Decisions started as one file per ADR in `docs/decisions/`. You asked for a single place to
read every decision.

### Options
1. Keep separate files **and** add a combined file: two copies that will drift apart.
2. **One file (`docs/DECISIONS.md`) replaces the folder.**
3. Keep separate files with a longer index: still means clicking through many files.

### Decision
Option 2. The old per-ADR files were removed; they remain in git history (commit `bded884`).

### Consequences
- ✅ One file to read, search and review in a PR.
- ✅ No duplicate copies to keep in sync.
- ⚠️ The file will grow. If it ever gets unwieldy (30+ ADRs), we can split by phase with a new ADR.
