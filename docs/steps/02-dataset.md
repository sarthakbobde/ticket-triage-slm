# Step 02: Pick & explore the dataset

**Who:** 🤝 (Claude wrote the script, you run it and read the data) · **Status:** in progress · **Cost:** $0

## Decisions made in this step
- [ADR-007](../DECISIONS.md#adr-007): dataset = Tobi-Bueck/customer-support-tickets, English only
- [ADR-008](../DECISIONS.md#adr-008): categories = dataset's top queues (finalised after you run the script)

## 1. Run the script
```bash
uv run python scripts/01_download_data.py
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
_Fill in from the notebook's Summary section._

## What you learn
Data profiling, stable ids, leakage, label noise, and why the label set is designed from data,
not imagination.
