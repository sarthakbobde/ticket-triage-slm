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
| [005](#adr-005) | Teacher = Gemini; exact model chosen by a 50-ticket pilot | Accepted; amended: teacher is Azure `gpt-4.1` | 2026-10-29 |
| [006](#adr-006) | All decisions kept in one file | Accepted | 2026-10-29 |
| [007](#adr-007) | Dataset = Tobi-Bueck/customer-support-tickets, English only | Accepted | 2026-10-29 |
| [008](#adr-008) | ~~Categories = dataset's top queues; priority mapped 5 → 4 levels~~ | Superseded by ADR-010 | 2026-10-29 |
| [009](#adr-009) | EDA in notebooks, pipeline in scripts; notebook tools as uv dev deps | Accepted | 2026-10-29 |
| [010](#adr-010) | 6 merged categories, 3 priorities; truth = teacher + hand-labelled gold set | Accepted; amended: 5 categories (ADR-013) | 2026-10-30 |
| [011](#adr-011) | `entities` replaced by `products: list[str]` | Accepted | 2026-10-30 |
| [012](#adr-012) | Test set first (natural distribution), capped training sample, 200-ticket gold set | Accepted | 2026-10-30 |
| [013](#adr-013) | Merge `hr` into `customer_service` (5 categories, schema v1.1); mine extra rare-class tickets | Accepted | 2026-10-31 |
| [014](#adr-014) | Base model = Qwen2.5-3B-Instruct, chosen from an untrained baseline vs Llama 3.2 3B | Accepted | 2026-10-31 |
| [015](#adr-015) | Restructure into an installable package with one `triage` CLI, tests, lint and CI | Accepted | 2026-10-31 |
| [016](#adr-016) | Serve the fine-tune as Q4_K_M GGUF through Ollama | Accepted | 2026-11-01 |

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
**Status:** Accepted: Gemini (exact model chosen by pilot in Step 03) · **Date:** 2026-10-29 · **Step:** 03

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
**Gemini** (you already have a key). Prices checked on ai.google.dev on 2026-10-29 for ~3k
tickets (~1.5M input / ~0.9M output tokens):

| Model | Standard | Batch API (−50%) |
|---|---|---|
| Gemini 2.5 Flash-Lite | ~$0.50 | ~$0.25 |
| Gemini 3.5 Flash-Lite | ~$2.70 | ~$1.35 |
| Gemini 3.8 Flash (promo price until 2026-12-31, then 2×) | ~$4.50 | ~$2.25 |
| Gemini 3.1 Pro | ~$15–50 | ~$7–25 |
| Free tier | $0, rate-limited | — |

Process:
1. **Pilot:** label the same 50 tickets with two candidates (e.g. 3.8 Flash vs 3.5 Flash-Lite).
2. Compare each against the dataset's queue/priority labels and hand-review ~20.
3. Pick the cheaper model unless the stronger one is clearly better.
4. Label the full set on the free tier if its limits allow; otherwise use the Batch API.

Settings: **thinking budget low/off** (thinking tokens are billed as output), temperature 0,
JSON response mode with our schema.

**Terms note:** Google's API terms restrict using outputs to develop *competing* models. This is
a personal learning/portfolio project on a public synthetic dataset; we're not building a
commercial model. If this ever became commercial, revisit this ADR.

### Consequences
- ✅ Expected total cost under $5, possibly $0.
- ✅ Gemini exposes an OpenAI-compatible endpoint, so ADR-004 holds: config change only.
- ⚠️ Free-tier prompts may be used by Google to improve its products. Acceptable here (public
  synthetic data), never acceptable for real customer tickets.
- ⚠️ Promo prices change after 2026-11-30 / 2026-12-31. Re-check before the full run.

**Amendment (2026-10-31, Step 03):** the teacher is **Azure AI Foundry `gpt-4.1`**, not Gemini.
- Why: the user already has Foundry access, and it was the only chat deployment the key could
  reach (the other names we probed returned 404). It is a non-reasoning model, accepts
  `temperature: 0` and `json_schema`, and uses the same OpenAI-compatible client (ADR-004).
- Set in `configs/models.yaml` (endpoint `.../openai/v1/`, key in `.env` as `TEACHER_API_KEY`).
- Cost so far is about $3.7 at assumed prices of $2 in / $8 out per 1M tokens. **These prices are
  unverified**; confirm them in the Foundry portal.
- The Gemini price table above is kept as the record of what was considered.
- The Terms note applies to Google only. Check Azure OpenAI's terms on using outputs for
  training before this ever becomes more than a learning project.

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

---

<a id="adr-007"></a>
## ADR-007 — Dataset: Tobi-Bueck/customer-support-tickets, English only
**Status:** Accepted · **Date:** 2026-10-29 · **Step:** 02

### Context
We need realistic, email-length support tickets, ideally with some existing labels to
sanity-check the teacher. Candidates were checked on the Hugging Face Hub on 2026-10-29.

### Options
| Dataset | Rows | Shape | Licence | Verdict |
|---|---|---|---|---|
| **Tobi-Bueck/customer-support-tickets** | 61.7k (28.3k en) | subject + body (median ~415 chars) + agent answer; labels: queue, priority, type, tags | CC-BY-NC-4.0 | **chosen** |
| bitext/…customer-support-llm-chatbot | 26.9k | one-line templated chatbot messages | CDLA-Sharing-1.0 | not ticket-like |
| Console-AI/IT-helpdesk-synthetic-tickets | 500 | IT tickets | MIT | too small |

Language: English only vs English + German. German doubles eval work, and small models are
weaker in German.

### Decision
Tobi-Bueck, **English only**. We'll sample ~3k tickets for labelling in Step 03.

### Consequences
- ✅ Existing `queue`/`priority` labels give a free agreement check for the teacher and the student.
- ⚠️ **Synthetic data with visible label noise** (e.g. a product question filed under
  "Returns and Exchanges"). Dataset labels are a *signal*, not ground truth.
- ⚠️ Multiple dataset versions are merged (`version` 51 / 400), and ~13k rows lack `type`.
- ⚠️ **Non-commercial licence**: fine for a portfolio, not for a commercial product.
  Credit the dataset in the README.

---

<a id="adr-008"></a>
## ADR-008 — Category list and priority levels (schema v1.0)
**Status:** Superseded by [ADR-010](#adr-010): the EDA showed the dataset's queues are too noisy to adopt as-is · **Date:** 2026-10-29 · **Step:** 02

### Context
The draft schema (ADR-001) had e-commerce categories (refund, shipping…), but the dataset's
queues are IT/SaaS-flavoured. Priority has 5 levels in the dataset vs 4 in our schema.

### Options
1. Keep the draft categories: the teacher labels everything, and dataset queues can't be compared.
2. **Adopt the dataset's top queues** and merge the long tail into `other`.
3. Invent a new taxonomy: same downside as 1.

### Decision
Option 2. The exact list is filled in here after profiling (6–8 queues, minimum ~2% share each,
possibly merging queues a human can't tell apart).

Priority mapping (proposed):

| Dataset | Ours |
|---|---|
| very_low, low | low |
| medium | medium |
| high | high |
| critical | urgent |

Final category list: _TBD after profiling._

### Consequences
- ✅ Teacher and student can both be scored against an independent label source.
- ✅ Bumps `SCHEMA_VERSION` to 1.0 and fixes the schema before any money is spent on labelling.
- ⚠️ Merging `very_low` into `low` loses detail, but the two are rarely actionable differently.

---

<a id="adr-009"></a>
## ADR-009 — Notebooks for exploration, scripts for the pipeline
**Status:** Accepted · **Date:** 2026-10-29 · **Step:** 02

### Context
You want to do EDA interactively. Notebooks are great for exploring but bad for repeatable
pipelines (hidden state, out-of-order cells, hard to diff or review).

### Options
1. Everything in notebooks.
2. Everything in scripts.
3. **Notebooks for exploration (`notebooks/`), scripts for anything the pipeline depends on (`scripts/`).**

Dependencies for notebooks:
- a. Add to core dependencies: but they're not needed to *run* the product.
- b. Optional extra (`--extra eda`): easy to forget, and `uv sync` without it removes them.
- c. **uv dev-dependencies**: installed by default with `uv sync`, excluded from the package.

### Decision
Option 3 + c. `ipykernel`, `matplotlib` and `scikit-learn` are dev dependencies. scikit-learn
is also used for the TF-IDF baseline and later for eval metrics (F1, confusion matrix).

Rule: if a notebook result feeds the pipeline (e.g. a filter or a mapping), move that logic
into a script or `src/triage/` and record the decision here.

### Consequences
- ✅ Free exploration without polluting the pipeline.
- ✅ The TF-IDF + logistic regression baseline gives a **non-LLM benchmark** for `category`/`priority`.
- ⚠️ Notebooks with outputs make noisy git diffs. Clear outputs before committing, or accept it.

---

<a id="adr-010"></a>
## ADR-010 — Six categories; truth = teacher guidelines + gold set
**Status:** Accepted · **Date:** 2026-10-30 · **Step:** 02 · **Supersedes:** ADR-008

### Context
EDA (`notebooks/02_eda.ipynb`) on 23,618 English tickets:
- A TF-IDF + logistic regression baseline scores **macro-F1 0.58** on the 10 dataset queues.
- Queue-profile similarity clusters {IT Support, Technical Support} (0.57) and
  {Customer Service, Sales, General Inquiry, Returns} (0.36–0.70).
- Reading samples: near-identical tickets ("security protocols for medical data") appear under
  Technical, IT, Product **and** Customer Service. Priority is similarly inconsistent (a data
  breach labelled `low`, a best-practices question labelled `high`); baseline macro-F1 0.63.
- English data has only low / medium / high priorities.

**Conclusion:** the dataset's labels are only weakly determined by the text. That's label
noise, and no model can learn it away.

Baseline with merged queues (5-fold CV):

| Option | Classes | Macro-F1 | Acc | Majority share |
|---|---|---|---|---|
| A: original | 10 | 0.58 | 0.56 | 0.29 |
| B: IT+Tech, CS+Sales+General | 7 | 0.63 | 0.63 | 0.41 |
| **C: B + Returns→CS** | **6** | **0.66** | **0.65** | 0.41 |
| D: C + Product→Tech | 5 | 0.70 | 0.74 | 0.59 |

### Options
- Category set: B, C or D. D scores highest but "technical" becomes 59% of tickets, which is
  useless for routing.
- Truth source: (1) dataset labels as targets: model learns the noise, ceiling ≈ 0.65;
  (2) **written definitions + teacher labels + human gold set**.

### Decision
- **Categories (C):** `technical`, `product`, `billing`, `outage`, `customer_service`, `hr`.
  The split between technical and product is defined by **"is something broken?"**, a rule a
  human can apply from the text alone.
- **Priority:** `low`, `medium`, `high` (no `urgent`; it doesn't exist in the English data).
- **Truth = labelling guidelines** in `src/triage/prompts.py` (`CATEGORY_DEFINITIONS` +
  priority rubric), applied by the teacher, and audited by a **hand-labelled gold set** (ADR-012).
- Dataset labels are mapped via `src/triage/dataset_map.py` and used **only as a cross-check**
  (teacher–dataset agreement), never as targets.

### Consequences
- ✅ Labels follow rules a human can check, so scores are meaningful.
- ✅ Mirrors industry practice: business defines the taxonomy, annotators follow guidelines,
  gold set keeps them honest.
- ✅ The 0.66 TF-IDF baseline is the **number to beat** on the merged dataset labels.
- ⚠️ Teacher–dataset agreement will be modest (~0.6–0.7). That's expected, not a bug.
- ⚠️ Changing `CATEGORY_DEFINITIONS` changes the truth: new ADR + `SCHEMA_VERSION` bump + relabel.

**Amendment (2026-10-31, Step 03):** the category set is now **five**, see ADR-013. `hr` is
folded into `customer_service`. The rest of this ADR (truth = written guidelines + teacher +
gold set; dataset labels only as a cross-check) is unchanged. The measured teacher–dataset
agreement is only **37%**, lower than the 0.6–0.7 expected above, because the dataset's queue
labels are largely unrelated to the ticket text; the teacher follows the guidelines.

---

<a id="adr-011"></a>
## ADR-011 — Replace `entities` with `products`
**Status:** Accepted · **Date:** 2026-10-30 · **Step:** 02 · **Amends:** ADR-001 (entities row)

### Context
EDA: only **3.4%** of tickets contain any anonymisation placeholder (`<tel_num>` 288,
`<acc_num>` 64, `<email>` 3). Order ids, emails and amounts are effectively absent, so those
fields would be null ~97%+ of the time and the model would learn "always null".
`<br>` (2,521) is HTML, not an entity. What tickets *do* mention constantly is named
products/software (Elasticsearch, Jira, MacBook Pro, Zoho Books, Symantec…).

### Options
1. Keep the 4 fields: mostly null, so useless.
2. **`products: list[str]` only.**
3. `products` + an `issue_type` enum: overlaps with category and the dataset's tags; more to label.

### Decision
Option 2. `products` lists names explicitly in the ticket; `[]` if none; never invented.
Evaluated with set precision/recall per ticket (Step 05).

### Consequences
- ✅ A field that is actually populated and useful for routing to product specialists.
- ⚠️ No ground truth in the dataset; teacher output + gold set only.
- ⚠️ Text cleaning needed before labelling: `<br>` → newline, literal `
` escapes → newline (Step 03 script).

---

<a id="adr-012"></a>
## ADR-012 — Sampling, splits and the gold set
**Status:** Accepted · **Date:** 2026-10-30 · **Step:** 02 → used in 03/04

### Context
Mapped category shares: technical 40.9%, customer_service 24.4%, product 18.6%, billing 10.2%,
outage 4.0%, hr 1.9%. Random sampling would give the small model very few `hr`/`outage` examples.
The first idea (cap every queue at 340, the size of General Inquiry) would consume *all* of
General Inquiry, leaving nothing for testing, and makes the test set unrealistically balanced.

### Decision
Order matters:
1. **Near-duplicate removal** on normalised text (lowercase, collapse whitespace, strip
   punctuation) *before* any split, so copies can't straddle train and test.
2. **Test set first:** 500 tickets, **natural distribution** (stratified on mapped category).
   It measures real-world performance and is frozen forever.
3. **Validation:** 300 tickets, natural distribution.
4. **Train pool:** from the rest, **capped per mapped category**, so rare classes are
   over-represented relative to reality but not wildly. Target ~2,500 tickets; the cap is
   derived from that target (**416** per category, see `docs/reports/splits.md`).
5. **Gold set:** 200 of the 500 test tickets, **hand-labelled by you** with the same guidelines,
   *before* looking at the teacher's labels (to avoid anchoring).

All splits use `SEED = 42`. Total to label with the teacher: ~3,300 tickets (cost unchanged, ADR-005).

### Consequences
- ✅ Honest test numbers + enough rare-class training examples.
- ✅ Gold set lets us measure the teacher itself, i.e. the ceiling for the student.
- ⚠️ `hr` in a 200-ticket gold set is only ~4 tickets; report per-class results with counts.
- ⚠️ Stratifying on noisy dataset labels is approximate, but good enough for sampling.

**Amendment (2026-10-30, Step 03):**
- The first draft said "cap ≈ 600", but 6 × 600 overshoots the ~2,500 target (and the
  ADR-005 budget). The target wins; the cap is computed from it.
- Splits are built **before** labelling (they decide what we pay to label), so the scripts are
  reordered: `02_build_splits.py` → `03_label_with_teacher.py`.
- The test set is labelled by the teacher only **after** you hand-label the gold set (Step 04),
  so you can't anchor on the teacher's answers even by accident.
- **(2026-10-31)** With five categories (ADR-013) the per-category cap no longer applies to a
  separate `hr` class. The splits were built under schema v1.0 and are **frozen**: do not re-run
  `02_build_splits.py`, because `dataset_map.py` now maps Human Resources to `customer_service`
  and the script would draw a different sample.

---

<a id="adr-013"></a>
## ADR-013 — Merge `hr` into `customer_service`; mine extra rare-class tickets
**Status:** Accepted · **Date:** 2026-10-31 · **Step:** 03 · **Amends:** ADR-010 (category set), ADR-012

### Context
After labelling 1,000 training tickets the teacher found only **1.4% `hr`** and **3.3%
`customer_service`**. Two rounds of keyword mining over the unused pool added examples:
round 1 picked 500 tickets (200 for hr, 300 for customer_service), round 2 a stricter hr search
(40 tickets). Result: `customer_service` reached 115 examples, but `hr` stopped at **38**.
The pool is exhausted: about 41 genuine HR tickets exist in 21,816 unused tickets, and 38–39 were
found. The dataset's "Human Resources" queue is mostly not HR (about 9% of it is). In the
test set that leaves roughly **one** genuine hr ticket, so a score for `hr` would be meaningless
and one miss would move macro-F1 by several points. Details: `notebooks/03_teacher_labels.ipynb`.

### Options
1. Keep six categories and live with 38 training / ~1 test example.
2. Delete the 38 hr tickets and the category.
3. **Merge `hr` into `customer_service`** and widen its definition.

### Decision
Option 3. Schema **v1.1** has five categories: `technical`, `product`, `billing`, `outage`,
`customer_service`. The `customer_service` definition now includes internal employee matters
(payroll, onboarding, leave, staff policies). `Human Resources` in `dataset_map.py` maps to
`customer_service`.
- The teacher labelled with the six-category prompt. **It is not called again.**
  `scripts/03_merge_hr.py` copies the labels and changes `hr` to `customer_service`
  (39 rows: val 1, train 14, train_extra 20, train_extra2 4). Raw files are untouched; merged
  files go to `data/labeled/merged/` and keep `merged_from: "hr"`.
- Training-set counts after the merge (1,540 labelled): product 692, technical 448,
  customer_service 153, billing 142, outage 105.

### Consequences
- ✅ Every category has enough examples to learn and to report a score.
- ✅ No tickets are thrown away.
- ⚠️ Mining was biased towards rare classes, so the training mix is not the natural mix. The
  test set stays natural. Priority is skewed (medium only 20%): check against the gold set.
- ⚠️ `customer_service` is now a mixed bucket. If HR routing ever matters, it needs its own
  data first.
- ⚠️ Any later change to the definitions needs a new ADR and a version bump again.

---

<a id="adr-014"></a>
## ADR-014 — Base model: Qwen2.5-3B-Instruct
**Status:** Accepted · **Date:** 2026-10-31 · **Step:** 06 · **Completes:** ADR-002 (family chosen with data)

### Context
ADR-002 fixed the size (1.5B–3B) and said to baseline two untrained candidates. Both were run
through the Step 05 harness on the 200 gold tickets, served by Ollama (4-bit, GTX 1650),
once with a plain prompt and once with schema-enforced JSON. Gold labels were drafted by Claude
and accepted by the user, so scores measure agreement with those labels. Reports: `evals/reports/`.

| | Qwen plain | Llama plain | Qwen schema | Llama schema |
|---|---|---|---|---|
| Valid JSON | 100% | 2% | 100% | 100% |
| Category accuracy / macro-F1 | 75.5 / 62.5 | 2.0 / 2.0 | 75.5 / 62.5 | 71.0 / 68.0 |
| Priority accuracy | 40.5 | 1.5 | 40.5 | 76.0 |
| Sentiment accuracy | 63.0 | 1.0 | 63.0 | 59.0 |
| Products F1 | 36.3 | 2.0 | 65.0 | 64.2 |
| Model size in Ollama | 1.9 GB, fully on GPU | 2.9 GB, 20% spills to CPU | | |

Notes: Llama's plain-prompt answers lose the closing `}` (170 of 200 become valid if it is
added) or arrive in code fences (30 of 200); the harness stays strict. Latencies are rough: the
last 90 Llama schema tickets ran after a server restart with part of the model on the CPU.
Both models are weak on `customer_service` and `outage` untrained.

### Options
1. Llama 3.2 3B: better priority and macro-F1 untrained, but bigger, spills out of 4 GB VRAM
   at 4k context, and needs schema enforcement just to be valid.
2. **Qwen2.5-3B-Instruct:** smaller, fits the GPU, valid JSON without help.
3. Qwen2.5-1.5B (Apache-2.0 per our recollection; not baselined): smaller still, a fallback.

### Decision
Fine-tune **Qwen2.5-3B-Instruct** (Ollama tag `qwen2.5:3b-instruct`, Hugging Face
`Qwen/Qwen2.5-3B-Instruct`). Untrained gaps matter less than fit: its weak fields (priority,
`customer_service`) are what the training data teaches, and it must serve fully on the GPU in Step 10.

### Consequences
- ✅ Fits the 4 GB card with room for context; valid JSON out of the box.
- ✅ **Numbers to beat** (Qwen schema-enforced): category accuracy 75.5%, macro-F1 62.5%,
  priority 40.5%, sentiment 63.0%, products F1 65.0%.
- ⚠️ **Licence:** Hugging Face lists the 3B model as `qwen-research`, not Apache-2.0. The
  terms were not read (the page does not spell out restrictions); read the LICENSE file in the
  repo. Fine for a learning/portfolio project; if this ever became commercial, switch to
  Qwen2.5-1.5B (check its licence) or revisit this ADR.
- ⚠️ Priority baselines differ a lot between models (40% vs 76%), so a fine-tune that only reaches
  Llama's untrained 76% would not be a win. Compare per field, not only on category.
- ⚠️ The gold set is small and its labels came from Claude: a few points either way is noise.

---

<a id="adr-015"></a>
## ADR-015 — Package layout, one CLI, tests and CI
**Status:** Accepted · **Date:** 2026-10-31 · **Step:** cross-cutting (after Step 07)

### Context
The pipeline had grown as nine numbered scripts. They worked, but: the same helpers were
copied between them (`read_jsonl` five times, the client setup twice), paths and options were
module-level constants edited by hand (mining's `ROUND`), logic could only be run, not tested,
and nothing checked style or ran tests automatically. That is fine for exploring and wrong for
code other people will read, change and trust.

### Options
1. Keep scripts, add tests around them. Scripts are hard to import and test, and duplication stays.
2. **Installable package with feature modules, one `triage` command, tests, ruff, CI.**
3. A workflow orchestrator (Airflow, Prefect, DVC). Overkill for a linear 10-step pipeline run by one person.

### Decision
Option 2. Layout and rules are in [ARCHITECTURE.md](ARCHITECTURE.md). The command line is
`argparse` (standard library, already understood) rather than a new dependency. Code stays in
plain loops and named steps; only the structure changed, and the new commands reproduce the old
outputs byte for byte (checked on the merged labels, training files, gold sheet and all eval reports).

Old command → new command:

| Was | Is now |
|---|---|
| `scripts/01_download_data.py` | `triage data download` |
| `scripts/02_build_splits.py` | `triage data splits` |
| `scripts/03_label_with_teacher.py` | `triage label` |
| `scripts/03_mine_rare_classes.py` (edit `ROUND`) | `triage data mine --round N` |
| `scripts/03_merge_hr.py` | `triage data merge-hr` |
| `scripts/03_pilot_report.py` | `triage pilot-report` |
| `scripts/04_make_gold_sheet.py` | `triage data gold-sheet` |
| `scripts/07_prepare_training_data.py` | `triage data training-data` |
| `evals/run_model.py` | `triage eval run` |
| `evals/score.py` | `triage eval score` |

Behaviour changes made on purpose: `triage label` now also needs `--yes` for remote (paid)
endpoints, like `triage eval run`; unknown keys in `models.yaml` are errors; metrics use
`zip(strict=True)` so a length mismatch cannot silently drop tickets; the pilot report no
longer imports scikit-learn (a dev-only dependency) and macro-F1 there skips classes with no
true tickets, as the eval harness does.

### Consequences
- ✅ Logic is importable and tested (41 tests, no network or GPU); lint and tests run in CI.
- ✅ One definition of paths, config loading, client setup and jsonl I/O.
- ✅ Safer by default: paid calls are opt-in on the command line.
- ⚠️ Older ADRs, step guides written before this change and the commit history still name the
  old scripts; the table above is the translation. Guides were updated to the new commands.
- ⚠️ `triage data splits` must not be re-run: it would draw a different sample (ADR-012 amendment).
- ⚠️ More files and indirection than a script. The cost is worth it only because the project
  keeps growing (training, quantisation, serving); a throwaway analysis should stay a notebook.

---

<a id="adr-016"></a>
## ADR-016 — Serve the fine-tune as Q4_K_M GGUF through Ollama
**Status:** Accepted · **Date:** 2026-11-01 · **Step:** 09

### Context
The fine-tuned Qwen2.5-3B (ADR-014, Step 07) was merged and converted to GGUF at three compression
levels, each scored on the 200 gold tickets on the GTX 1650 (plugged in). Details: `docs/steps/09-gguf-ollama.md`.

### Decision
Serve **Q4_K_M** (`ticket-triage-q4`, 1.9 GB) through Ollama.
- Quality: within noise of Q8_0 and Q5_K_M (category 77.5% vs 78.5% / 79.5%; sentiment and products on par).
- Speed: p50 2.3 s vs 2.7 s (Q5) and 6.3 s (Q8, which spills onto the CPU).
- Size: runs 100% on the 4 GB GPU with room for the context.
Q5_K_M is the fallback if a later, larger gold set shows Q4 losing accuracy.

### Consequences
- ✅ The local model matches the teacher on this gold set (category 77.5-79.5% for both) at no per-ticket cost.
- ⚠️ A single request is slower locally (2.3 s) than the Azure teacher (1.3 s): the case for the small model is cost, privacy and control, not latency on this GPU.
- ⚠️ 200 gold tickets cannot separate quantization levels that differ by a few points.
- ⚠️ Remaining label disagreements (`customer_service`, priority) are inherited from the teacher (Step 08 finding).
- ⚠️ Ollama 0.40 imports only finished GGUF files; conversion needs llama.cpp's tools (b12024).
