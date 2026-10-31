# Step 02: Pick & explore the dataset

**Who:** 🤝 (Claude wrote the script, you run it and read the data) · **Status:** in progress · **Cost:** $0

## Decisions made in this step
- [ADR-007](../DECISIONS.md#adr-007): dataset = Tobi-Bueck/customer-support-tickets, English only
- [ADR-008](../DECISIONS.md#adr-008): categories = dataset's top queues (finalised after you run the script)

## 1. Run the script
```bash
uv run triage data download
```
The first run downloads the dataset into `E:\hf_cache` (tens of MB). If you see a
`ResourceTracker ... _recursion_count` error at the very end, ignore it: it's a known bug in
Python 3.12.0 and happens after everything has been written.

It produces:
- `data/raw/tickets_en.jsonl`: the cleaned English tickets (git-ignored)
- `docs/reports/dataset-profile.md`: counts and distributions (commit this)

## 2. What the script does, and why

| Stage | Why |
|---|---|
| Keep `language == "en"` | ADR-007: one language keeps evals simple |
| `text = subject + body` | That's what an agent sees; the model should see the same |
| Drop bodies < 30 chars | Nothing to triage |
| `id = sha1(text)[:12]` | **Stable ids**: row numbers change if upstream re-orders; a content hash doesn't. Same text → same id, so duplicates collapse. |
| Drop exact duplicates | A duplicate in both train and test inflates scores (**leakage**). Near-duplicates are handled in Step 04. |
| Collapse `tag_1..tag_8` → `tags` list | Easier to work with; maybe useful features later |
| Random examples with `SEED=42` | Same 10 tickets every run, so we can discuss them |

## 3. Your task: read the data like a scientist
Open `docs/reports/dataset-profile.md` and look at the 10 printed tickets. Answer these in the
**Findings** section below (a sentence each is fine):

1. **Queues:** which ~6–8 queues cover most tickets? Which tiny ones should merge into `other`?
2. **Overlap:** do "Technical Support", "IT Support" and "Product Support" look genuinely
   different in the examples? Could *you* tell them apart from the text alone? If a human can't,
   a model can't either, so they might need merging.
3. **Priority:** look at the queue × priority table. Is priority spread sensibly, or is almost
   everything "medium/high"? (A skewed label means accuracy alone will be misleading; that's why
   we'll also use macro-F1.)
4. **Label noise:** in the 10 examples, how many labels do you disagree with?
5. **Versions:** how many dataset versions are there, and do rows with missing `type` come from
   one version?

## 4. EDA notebook
Two versions with the same questions:
- `notebooks/02_eda.ipynb`: **work here**. Hints and spoilers are folded; click to reveal.
- `notebooks/02_eda_hints_visible.ipynb`: reference copy with hints and early findings shown inline.

Open `notebooks/02_eda.ipynb` in VS Code and select the `.venv` Python kernel (top right:
"Select Kernel" -> Python Environments -> `.venv`). Prefer the browser? Run
`uv run --with jupyterlab jupyter lab`.

Each section answers one question that **changes a decision**. Write the implication under
each section, then copy the summary below. Decision record: [ADR-009](../DECISIONS.md#adr-009).

<details><summary>👀 Spoiler: Claude's early signals from the profile report (do the EDA first!)</summary>

- **16% exact duplicates** (4,497) were removed, so expect near-duplicates too (leakage risk).
- **English has only 3 priorities** (low/medium/high); no critical/very_low. Our schema has 4.
- **10 queues**, smallest 1.4%; the top 4 may overlap (Technical / Product / IT / Customer Service).
- **Priority is strongly tied to queue** (Outages 71% high, General Inquiry 59% low): a
  shortcut a model can learn without reading the ticket.
- Text is short: p90 ≈ 160 tokens.

</details>

## 5. Findings
_From the notebook's outputs; each item links to the decision it feeds._
1. **Near-duplicates (section 2):** 4,497 exact duplicates (16%) were already removed, and
   template-like tickets still repeat with small wording changes (e.g. "Request for Information
   on Medical Data Security"). So we remove near-duplicates on normalised text *before* any split
   ([ADR-012](../DECISIONS.md#adr-012)).
2. **Max sequence length (section 3):** ticket p99 = 1,003 chars ≈ 250 tokens (≈4 chars/token).
   250 + ~350 prompt + ~300 output ≈ 900, so `max_seq_length = 1024`.
3. **Sampling strategy (section 4):** mapped shares run from technical 40.9% down to hr 1.9%.
   Test (500) and val (300) keep the natural distribution; the train pool is capped at ≈600 per
   category (~2,500 total). A cap of 340 (General Inquiry's size) was rejected: it would use up
   all of General Inquiry, leaving none for testing ([ADR-012](../DECISIONS.md#adr-012)).
4. **Final categories (section 5, confusable queues):** IT / Technical and Customer Service /
   Sales / General Inquiry / Returns can't be told apart from the text
   ([queue similarity](../reports/queue-similarity.png); TF-IDF macro-F1 only 0.58 on 10 queues). Merged to 6:
   `technical`, `product`, `billing`, `outage`, `customer_service`, `hr`; macro-F1 0.66
   ([ADR-010](../DECISIONS.md#adr-010)).
5. **Priority levels:** `low`, `medium`, `high` only (no `urgent` in English data). Priority is
   inconsistent with the text and strongly tied to the queue, so it follows the written rubric,
   not the dataset labels ([ADR-010](../DECISIONS.md#adr-010)).
6. **Entity fields:** only 3.4% of tickets contain a placeholder (`<tel_num>` 288, `<acc_num>` 64,
   `<email>` 3, others ≤ 18); `<br>` (2,521) is HTML, not an entity. Product names are what
   tickets actually mention, so `entities` → `products: list[str]`
   ([ADR-011](../DECISIONS.md#adr-011)).
7. **Reply length:** dataset answers have a median of 373 chars (p75 519), so `suggested_reply`
   is targeted at ~350 chars (≈90 tokens), well within the ~300-token output budget.
8. **Versions (section 9):** 4 groups: 400 (44%), missing (31%), 52 (22.5%), 51 (2.3%). `type`
   has no missing values. Queue shares are nearly identical across versions (no queue differs by more than 3.3 pts), so
   we keep all versions and don't filter.

## 6. Decisions that came out of the EDA
- [ADR-008](../DECISIONS.md#adr-008) **superseded**: the dataset's queues are too noisy to adopt as-is.
- [ADR-010](../DECISIONS.md#adr-010): 6 categories, 3 priorities; truth = written guidelines +
  teacher + your gold set. Baseline to beat: macro-F1 **0.66** on merged dataset labels.
- [ADR-011](../DECISIONS.md#adr-011): `entities` → `products: list[str]`.
- [ADR-012](../DECISIONS.md#adr-012): near-dedupe → test (500, natural) → val (300) → capped
  train (~2,500); 200 test tickets hand-labelled by you.
- Code: `src/triage/schema.py` is now **v1.0**; category definitions in `src/triage/prompts.py`;
  queue mapping in `src/triage/dataset_map.py`.

## Remaining to close Step 02 (🧑)
- [x] Findings item 2: max sequence length (p99 ticket tokens + ~350 prompt + ~300 output, rounded up to a power of 2)
- [x] Findings item 1: split into "near-duplicates" (section 2) and "confusable queues" (section 5)
- [x] Findings item 8: versions (your crosstab answers it)
- [x] Move `seaborn` from core `dependencies` to `dev-dependencies` in `pyproject.toml`, then `uv sync`
- [x] Optional: move `docs/steps/image.png` → `docs/reports/queue-similarity.png` and update the link
- [ ] Commit

## What you learn
Data profiling, stable ids, leakage, label noise, and why the label set is designed from data,
not imagination.
