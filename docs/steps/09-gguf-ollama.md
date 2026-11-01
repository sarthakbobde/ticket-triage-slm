# Step 09: Merge, convert to GGUF, quantize, serve with Ollama

**Who:** 🤝 · **Status:** done · **Cost:** $0

## The idea
Training produced a small LoRA adapter (120 MB). To serve it locally we build one ordinary model
from it, change its file format, and shrink it so it fits the 4 GB GTX 1650:

```
adapter + base model --merge--> Hugging Face folder (6.2 GB, bf16)
   --convert_hf_to_gguf--> f16 GGUF (6.2 GB)  --llama-quantize--> Q8_0 / Q5_K_M / Q4_K_M
   --ollama create--> models served at localhost:11434
```

## Commands
```bash
uv sync --extra serve --extra export                  # torch (CPU), transformers, peft
uv run triage export merge                            # adapter + base -> models/merged/qwen-triage-v1
# llama.cpp tools (release b12024, downloaded into models/tools/, git-ignored):
.venv/Scripts/python.exe models/tools/llama.cpp-b12024/convert_hf_to_gguf.py models/merged/qwen-triage-v1 --outtype f16 --outfile models/gguf/qwen-triage-v1-f16.gguf
./models/tools/llama-bin/llama-quantize.exe models/gguf/qwen-triage-v1-f16.gguf models/gguf/qwen-triage-v1-Q4_K_M.gguf Q4_K_M
uv run triage export ollama --gguf models/gguf/qwen-triage-v1-Q4_K_M.gguf --name ticket-triage-q4
uv run triage eval run --config student_q4 --run ft-q4-local --format none
uv run triage eval score --run ft-q4-local
```
(`student_q5` and `student_q8` are the other two levels in `configs/models.yaml`.)

## Result (200 gold tickets, plain prompting, laptop plugged in)
| Metric | Kaggle fine-tune | Q8_0 3.3 GB | Q5_K_M 2.2 GB | Q4_K_M 1.9 GB | Teacher |
|---|---|---|---|---|---|
| Valid JSON | 100% | 100% | 100% | 100% | 100% |
| Category accuracy | 79.5% | 78.5% | 79.5% | 77.5% | 79.5% |
| Category macro-F1 | 65.7% | 62.4% | 66.4% | 60.9% | 66.4% |
| Priority accuracy | 70.0% | 72.0% | 70.0% | 67.0% | 68.5% |
| Sentiment accuracy | 72.0% | 79.5% | 76.0% | 80.0% | 70.0% |
| Products F1 | 88.0% | 84.1% | 85.4% | 84.5% | 85.0% |
| Latency p50 / p95 (s) | 2.0 / 2.4 (Kaggle T4) | **6.3 / 7.7** | 2.7 / 3.5 | **2.3 / 3.0** | 1.3 / 1.7 (Azure) |

- Differences of 1-5 points are 2-10 tickets: **noise**. Shrinking to 4 bits costs no measurable quality.
- Merge and conversion worked: every local variant is within a few points of the Kaggle run.
- Q8_0 is the slowest: 3.3 GB plus cache does not fit 4 GB, so Ollama ran it 40% CPU / 60% GPU.
  Q4_K_M and Q5_K_M run 100% on the GPU.

## What went wrong
1. Ollama 0.40 could not import the merged folder ("unsupported MLX architecture") or the adapter
   ("LoRA adapters are no longer supported"), and its `--quantize` accepts only `int4/int8/...`.
   Fix: llama.cpp's converter and quantizer, then Ollama imports the finished GGUF.
2. A relative path in the Modelfile was read as a model name to download. Use full paths.
3. On battery the GPU is power-capped: 10 tokens/s instead of 50. **Plug in before measuring latency.**
4. Windows Task Manager's default GPU graph shows 3D load, not CUDA compute, and on laptops "GPU 0"
   is often the integrated GPU. To check that Ollama uses the NVIDIA card, run `nvidia-smi` or
   `ollama ps` (the PROCESSOR column) while a request is running.
