# ADR-002 — Base model size

**Status:** Accepted (size). Exact model family chosen with data in Step 06.

## Context
- Local GPU: GTX 1650, **4 GB VRAM**, no tensor cores, no bf16.
- Training GPU: Kaggle T4, 16 GB.
- The product goal is "cheap and fast" — a big model defeats the purpose.

Rough memory maths (worth memorising):
- Weights ≈ parameters × bytes per parameter.
  - 3B in FP16 = 3 × 2 = **6 GB** → doesn't fit locally.
  - 3B in 4-bit ≈ 3 × 0.5 ≈ **1.8–2 GB** (+ overhead) → fits in 4 GB with room for context.
  - 7B in 4-bit ≈ **4–4.5 GB** → doesn't fully fit; needs CPU offload → slow.
- QLoRA training ≈ 4-bit weights + small LoRA adapters + activations → 3B trains comfortably on a T4.

## Options
| Size | Fits 1650 (Q4)? | Trains on T4? | Expected quality on narrow task |
|---|---|---|---|
| 0.5B–1B | easily | fast | ok for classification, weak replies |
| **1.5B–3B** | **yes** | **yes** | **good after fine-tuning** |
| 7B–8B | partial offload | tight | best, but slow locally |

## Decision
Target **1.5B–3B instruct models**. Candidate families (current at time of writing — verify on
Hugging Face before choosing): Qwen (1.5B/3B), Llama 3.2 3B, Gemma (2B-class), SmolLM.
We'll baseline 2 candidates untrained in Step 06 and fine-tune the better one.

## Consequences
- ✅ Whole loop (train → quantize → serve) works on free hardware.
- ✅ Makes the cost story dramatic vs. a frontier API model.
- ⚠️ `suggested_reply` quality will lag the teacher — that's expected and is why the gateway has a fallback.
- ⚠️ Licences differ by family (Llama/Gemma have custom licences; Qwen is mostly Apache-2.0). Check before choosing.
