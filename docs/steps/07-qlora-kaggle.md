# Step 07: Fine-tune Qwen2.5-3B with QLoRA on Kaggle

**Who:** 🤝 (Claude wrote the script and notebook; you run it on Kaggle) · **Status:** ready to run · **Cost:** $0

## Decisions used
- [ADR-014](../DECISIONS.md#adr-014): base = Qwen2.5-3B-Instruct
- [ADR-003](../DECISIONS.md#adr-003): train on Kaggle (free T4), serve locally
- [ADR-013](../DECISIONS.md#adr-013): five categories; training labels are the merged teacher labels

## The idea
The model already writes JSON, but it does not know our rules. We show it 1,540 examples of
`ticket -> teacher's JSON answer` and adjust a small **LoRA adapter** (about 1% of the model's size)
so its answers move towards the teacher's. The big model stays frozen and squeezed to 4-bit
(**QLoRA**), so it fits the T4's 16 GB.

```
laptop                              Kaggle (free T4)                        laptop
triage data training-data -> notebooks/07_train_qwen_kaggle.ipynb -> triage eval score
  data/training/*.jsonl  --upload-->  train, save adapter, answer 200 gold  --download--> Step 08
```

## 1. Prepare the data (free, 5 s)
```bash
uv run triage data training-data
```
Writes `data/training/` (git-ignored): `train.jsonl` (1,540), `val.jsonl` (300), `gold_prompts.jsonl` (200, **no labels**).
The script refuses to continue if any training ticket (or a copy of its text) is in the test or val set.

| Check | Result |
|---|---|
| Categories in train | product 692, technical 448, customer_service 153, billing 142, outage 105 |
| Longest example | about 950 tokens, so `MAX_LENGTH = 1280` is safe |
| Prompt | built with `build_messages()`, the same text the harness and the server use |

## 2. Run on Kaggle
1. Upload the 3 files as a **private** Kaggle Dataset called `ticket-triage-training`.
2. New Notebook -> import `notebooks/07_train_qwen_kaggle.ipynb` -> add the dataset as input ->
   Accelerator **GPU T4 x2**, Internet **On** -> **Run All** (roughly 30-60 min).
3. Download from the Output tab: `ft-qwen-kaggle.jsonl`, `qwen-triage-lora.zip`, `loss_curve.png`.

If a cell errors, paste the error back: the notebook was written against the current `trl`/`peft`
APIs but has not been run on Kaggle yet, and library versions drift.

## 3. Score it (free)
```bash
# put ft-qwen-kaggle.jsonl into evals/runs/ first
uv run triage eval score --run ft-qwen-kaggle
```
Compare with `evals/reports/base-qwen-schema.md` (that comparison is Step 08).

## Settings and why
| Setting | Value | Why |
|---|---|---|
| LoRA rank `r` / alpha | 16 / 32 | Standard recipe: enough capacity for a narrow task. |
| Layers adapted | all attention + feed-forward projections | Works better than attention alone. |
| Learning rate | 2e-4, cosine, 5% warm-up | Usual QLoRA value; decays smoothly. |
| Epochs | 3 | 1,540 examples is small; watch `eval_loss` and stop earlier if it rises. |
| Batch | 4 x 4 accumulation = 16 | Fits 16 GB with gradient checkpointing. |
| Loss counted on | the completion only | The model is graded on its JSON, not on repeating the prompt. |
| Precision | fp16 | The T4 has no bf16. |

## What to look at
1. `loss_curve.png`: both lines should fall. If training loss keeps falling while validation loss rises, it is memorising.
2. The "answers parse as JSON" count at the end of the notebook.
3. Then the real test in Step 08: per-field scores against the baseline numbers in ADR-014.

## Findings (fill in after the run)
1. Training time and final train / validation loss:
2. Valid JSON on the gold set:
3. Anything that broke on Kaggle:
