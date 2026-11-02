# Step 10: Serve the model locally

**Who:** 🧑/🤝 · **Status:** done · **Cost:** $0

## The idea
The fine-tuned model must answer requests from a program, not from a notebook. Ollama already does
the hard part: it loads the GGUF file onto the GPU and answers HTTP requests in the OpenAI format.
This step is about running it reliably and knowing what to expect from it.

```
your code / the gateway --HTTP--> Ollama (localhost:11434) --> ticket-triage-q4 on the GTX 1650
```

## Run it
```bash
E:/Ollama/ollama.exe list                 # should show ticket-triage-q4 (1.9 GB)
E:/Ollama/ollama.exe ps                   # while a request runs: PROCESSOR should say 100% GPU
```
The model files live in `E:\Ollama\models` (set by the `OLLAMA_MODELS` user variable). If Ollama
starts without that variable it looks in the wrong folder and finds no models; quit it and start it again.

## What to expect (measured on this laptop)
| Situation | Speed |
|---|---|
| Plugged in, model already loaded | about 50-55 tokens/s, about 2.4 s per ticket (p95 3.1 s) |
| **On battery** | about 8-10 tokens/s, 10-13 s per ticket: the GPU is power-capped |
| First request after 5 idle minutes | +7 s: Ollama unloads the model after 5 minutes (`ps` shows "UNTIL ...") |
| Q8_0 instead of Q4_K_M | slower (6.3 s): 3.3 GB plus cache spills onto the CPU |

Things that bit us, so they are written down:
- **Plug in before measuring or running a batch.** Battery mode looks exactly like a broken model.
- **Task Manager shows 0% GPU** for this work, because its default graph is the 3D engine. Use
  `nvidia-smi` or `ollama ps`, or switch a graph to "Cuda".
- **Keep it loaded:** set the user variable `OLLAMA_KEEP_ALIVE=-1` if the 7 s cold start matters.
- **Ollama 0.40 cannot import safetensors or LoRA adapters for Qwen2.** Only finished GGUF files (Step 09).

## Findings
1. Q4_K_M runs 100% on the GPU at about 2.4 s per ticket when plugged in.
2. A single request is slower here than the Azure teacher (about 1.3 s): the small model wins on cost and control, not on latency.
3. Next: put a gateway in front of it (Step 11).
