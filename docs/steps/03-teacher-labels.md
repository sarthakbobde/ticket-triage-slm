# Step 03: Split & label with a teacher model

**Who:** 🤝 (Claude wrote the scripts, you run them and judge the pilot) · **Status:** in progress · **Cost:** ≈ $0–5

## Decisions used in this step
- [ADR-005](../DECISIONS.md#adr-005): teacher = Gemini; the exact model is picked by a 50-ticket pilot
- [ADR-010](../DECISIONS.md#adr-010): labels follow the written guidelines in `src/triage/prompts.py`
- [ADR-012](../DECISIONS.md#adr-012): splits are built **before** labelling (see the 2026-10-30 amendment)

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
uv run python scripts/02_build_splits.py
```
Writes `data/processed/{test,val,train}.jsonl` and `docs/reports/splits.md` (commit the report).

| Stage | Why |
|---|---|
| Clean text: literal `\n` → newline, `<br>` → newline | ~88% of tickets contain literal `\n` escapes. The teacher and the student must see the **same** cleaned text. |
| Near-dedupe on normalised text | Copies straddling train and test inflate scores (leakage). Only 2 found: the exact-dup pass in Step 02 did most of the work. |
| Test 500 first, natural distribution | Measures real-world performance. **Frozen**: the script refuses to overwrite it. |
| 200 of test flagged `gold` | You hand-label these in Step 04 to measure the teacher itself. |
| Val 300, natural | Used for the pilot and, later, for choosing checkpoints. |
| Train ~2,500, capped at 416 per category | `hr` and `outage` get ~416 examples instead of ~50. |

**Your check:** open `docs/reports/splits.md`. Does every category have enough test tickets
to report a score? (Hint: look at `hr` in the gold column, ADR-012 warned about it.)

## 2. Set up the key and model ids
1. Put your Gemini key in `.env`: `TEACHER_API_KEY=...` (never commit it).
2. List the model ids your key can use:
   ```bash
   uv run python scripts/03_label_with_teacher.py --list-models
   ```
3. In `configs/models.yaml` → `teacher.candidates`, fix the three ids if they differ, and fill
   in their prices (USD per 1M tokens, ai.google.dev/pricing). Also set `rpm` to your tier's limit.

## 3. Pilot: same 50 tickets, three candidates (≈ $0.12)
```bash
uv run python scripts/03_label_with_teacher.py --split val --limit 50 --model gemini-3.8-flash
uv run python scripts/03_label_with_teacher.py --split val --limit 50 --model gemini-3.5-flash-lite
uv run python scripts/03_label_with_teacher.py --split val --limit 50 --model gemini-2.5-flash-lite
uv run python scripts/03_pilot_report.py
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

## 5. Label the full set (val + train)
```bash
uv run python scripts/03_label_with_teacher.py --split val
uv run python scripts/03_label_with_teacher.py --split train
```
About 2,800 calls. At 10 requests/minute that is roughly 5 hours; it's resumable, so run it in
the background and restart if it stops. Do **not** label `test` yet: you hand-label the gold
set first in Step 04, so the teacher's answers can't anchor you.

## 6. Findings
1. Splits (category counts look sensible?):
2. Pilot winner and why:
3. Failure rate (failed calls / tickets):
4. Actual cost of the full run:
5. Anything the teacher gets systematically wrong (update `prompts.py`? that's a new ADR):

## What you learn
Distillation, constrained decoding, schema validation as a contract, resumable and
cost-aware batch jobs, and judging a model by reading its outputs, not just by its scores.
