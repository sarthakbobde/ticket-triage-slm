# ADR-003 — Train in the cloud, serve locally

**Status:** Accepted

## Context
Local machine: GTX 1650 4 GB, Ryzen 5 5600H, 16 GB RAM, C: nearly full, E: 174 GB free.
Budget: free.

## Options
| Option | Train | Serve | Cost |
|---|---|---|---|
| All local | ❌ 4 GB too small, no bf16 | ✅ | free |
| **Kaggle train + local serve** | ✅ T4×2, 30 h/week | ✅ GGUF on 1650 | **free** |
| Colab train + local serve | ✅ T4, session limits | ✅ | free (less reliable) |
| Rented GPU (RunPod etc.) | ✅ fastest | ✅ | ~$0.2–0.5/h |

## Decision
- **Train on Kaggle** (Colab as backup).
- **Serve locally** with llama.cpp / Ollama using a quantized GGUF.
- Optionally run **vLLM on Kaggle** for a throughput benchmark — this contrasts "edge/laptop
  deployment" with "server deployment", which is a good talking point.

## Consequences
- ✅ Zero cost.
- ✅ You learn the real-world handoff: train in one environment, export, deploy in another.
- ⚠️ Training code must be a notebook that runs top-to-bottom on Kaggle with no local files
  except the uploaded dataset. Keep data in a Kaggle Dataset or on the HF Hub.
- ⚠️ Kaggle sessions time out (~12 h) — save checkpoints to `/kaggle/working` and push adapters to the HF Hub.
