# Step 03: Split & label with a teacher model

**Who:** 🤝 (Claude wrote the scripts, you run them and judge the pilot) · **Status:** labelling done; Step 04 next · **Cost:** ≈ $3.7 (assumed prices)

## Decisions used in this step
- [ADR-005](../DECISIONS.md#adr-005): teacher = Azure AI Foundry `gpt-4.1` (amended from Gemini)
- [ADR-010](../DECISIONS.md#adr-010): labels follow the written guidelines in `src/triage/prompts.py`
- [ADR-012](../DECISIONS.md#adr-012): splits are built **before** labelling (see the amendments)
- [ADR-013](../DECISIONS.md#adr-013): `hr` merged into `customer_service` (5 categories, schema v1.1)

## The idea: distillation
A large model (the **teacher**) labels ~3,300 tickets with our full JSON schema. The small model
(the **student**) is later trained to imitate those answers. So the teacher's quality is the
*ceiling* for the student, which is why we spend an afternoon choosing it.

```
tickets ─► 02_build_splits ─► test / val / train (unlabelled)
                                       │
           pilot: 50 val tickets × 3 candidates ─► 03_pilot_report ─► you pick a model
                                       │
                    03_label_with_teacher (val + train) ─► data/labeled/*.jsonl
```

## 1. Build the splits (free, ~10 s)
```bash
uv run triage data splits
```
Writes `data/processed/{test,val,train}.jsonl` and `docs/reports/splits.md` (commit the report).

| Stage | Why |
|---|---|
| Clean text: literal `\n` → newline, `<br>` → newline | ~88% of tickets contain literal `\n` escapes. The teacher and the student must see the **same** cleaned text. |
| Near-dedupe on normalised text | Copies straddling train and test inflate scores (leakage). Only 2 found: the exact-dup pass in Step 02 did most of the work. |
| Test 500 first, natural distribution | Measures real-world performance. **Frozen**: the script refuses to overwrite it. |
| 200 of test flagged `gold` | You hand-label these in Step 04 to measure the teacher itself. |
| Val 300, natural | Used for the pilot and, later, for choosing checkpoints. |
| Train ~2,500, capped at 416 per category | Rare classes get more examples than their natural share. |

**Your check:** open `docs/reports/splits.md`. Does every category have enough test tickets
to report a score? (Hint: `hr` had only 4 gold tickets. This is what led to ADR-013.)

## 2. Set up the key and model ids
1. Put your Foundry key in `.env` as `TEACHER_API_KEY=...` (never commit it, never put it in
   `models.yaml`; that file only holds the *name* of the variable).
2. `configs/models.yaml` → `teacher`: `base_url` is the Foundry `.../openai/v1/` endpoint and
   `model` is the **deployment name** (`gpt-4.1`).
3. Check the prices (USD per 1M tokens) in the Foundry portal. The values in the file are
   assumptions ($2 in / $8 out).
4. Every call to the teacher costs money: it is only run after you say yes.

## 3. Pilot: same 50 tickets (done with the Gemini candidates, kept as history)
```bash
uv run triage label --split val --limit 50 --model gemini-3.8-flash
uv run triage label --split val --limit 50 --model gemini-3.5-flash-lite
uv run triage label --split val --limit 50 --model gemini-2.5-flash-lite
uv run triage pilot-report
```
If a call fails with an error about `response_format` or the schema, set
`response_format: json_object` in `configs/models.yaml` and re-run. Validation still catches bad output.

**What the labelling script does, and why**

| Feature | Why |
|---|---|
| Same `SYSTEM_PROMPT` as training and serving | A student trained on one prompt and served with another silently gets worse. |
| `response_format: json_schema` | **Constrained decoding**: the API is told the exact JSON shape, so almost every answer parses. |
| Pydantic validation + 1 retry | The schema is the contract; anything that doesn't fit is logged to `failures__<model>.jsonl`, not saved. |
| Resumable (skips ids already written) | API calls cost money; a crash or Ctrl-C never pays twice. Just re-run the same command. |
| `temperature: 0`, `reasoning_effort: low` | Repeatable labels; thinking tokens are billed as output. |
| Tokens and latency saved per row | Feeds the cost report in Step 12. |

## 4. Your task: judge the pilot
Open `docs/reports/pilot.md`.
1. **Numbers:** agreement with the dataset labels around **0.6–0.7 is expected** (ADR-010: those
   labels are noisy). Look for big gaps between candidates, and for failed calls.
2. **Hand review:** read the 20 tickets (disagreements are listed first). For each, write
   which candidate you'd trust next to **Your call**, using the definitions in `prompts.py`.
   Also check `products` (nothing invented?) and `suggested_reply` (no promised refunds or timelines?).
3. **Decide:** pick the cheaper model unless the stronger one is *clearly* better on your review
   (ADR-005). Set `teacher.model` in `configs/models.yaml` and record the choice and the cost in ADR-005.

## 5. Label val and train
```bash
uv run triage label --split val
uv run triage label --split train
```
Resumable: re-running skips ids already written. Do **not** label `test` yet: you hand-label
the gold set first in Step 04, so the teacher's answers can't anchor you.

## 5b. Mine rare classes, then merge `hr` (what actually happened)
The first 1,000 training tickets had only 1% `hr` and 3% `customer_service`.
```bash
uv run triage data mine --round 2   # round 1 is already done
uv run triage label --split train_extra  # (round 2: train_extra2)
uv run triage data merge-hr   # free, no API call
```
Mining only *picks* tickets by keyword; the teacher decides the real category. Round 1 gave
115 `customer_service` but only 34 `hr`; a stricter round 2 reached 38 and the pool ran out
(about 41 genuine HR tickets exist). So ADR-013 merges `hr` into `customer_service`.
`03_merge_hr.py` writes `data/labeled/merged/*.jsonl`: **use these in Step 04 onwards**.
The full reasoning, with charts, is in `notebooks/03_teacher_labels.ipynb`.

## 6. Findings
1. Splits: test 500 (200 gold), val 300, train 2,496, cap 416. `hr` had 4 gold tickets.
2. Teacher: Azure `gpt-4.1` (the only deployment the key could reach).
3. Failure rate: 0 failed out of 1,840 labelled (val 300, train 1,000, mined 540).
4. Cost: about $3.7 at assumed prices; confirm in the portal.
5. Systematic issues: teacher–dataset agreement is only 37% because the dataset's queue labels
   are mostly unrelated to the text (ADR-010). Genuine `hr` is too rare to learn, so it was merged
   (ADR-013). Priority is skewed (medium only 20%): check it against the gold set in Step 04.

## What you learn
Distillation, constrained decoding, schema validation as a contract, resumable and
cost-aware batch jobs, and judging a model by reading its outputs, not just by its scores.
