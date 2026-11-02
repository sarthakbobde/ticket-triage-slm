"""Import a GGUF file into Ollama so it can be served and evaluated (Step 09).

The GGUF files come from llama.cpp's tools (see docs/steps/09-gguf-ollama.md):
    convert_hf_to_gguf.py   merged Hugging Face folder -> one 16-bit GGUF file (loses nothing)
    llama-quantize          16-bit GGUF -> smaller GGUF (Q8_0, Q5_K_M, Q4_K_M) (loses a little)

Ollama 0.40 cannot convert a Hugging Face folder or a LoRA adapter itself for this model
(it says "unsupported MLX architecture" and "LoRA adapters are no longer supported"), but it
imports a ready-made GGUF file without trouble.

    Q8_0     8 bits per number: close to the original, about 3.3 GB
    Q5_K_M   5 bits: about 2.2 GB
    Q4_K_M   4 bits: about 1.9 GB (fits the 4 GB card with room to spare)
"""

import logging
import subprocess
from pathlib import Path

from triage import paths
from triage.errors import UserError

log = logging.getLogger(__name__)

OLLAMA_EXE = Path("E:/Ollama/ollama.exe")

# The chat format Qwen was trained with (called ChatML). The training notebook built its examples
# with the same format, so serving must use it too: our messages are always a system + a user message.
TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""


def modelfile_text(gguf_file):
    """The Ollama recipe for the model: where the weights are, the chat format, and when to stop."""
    # Full path: Ollama reads a relative path as the name of a model to download.
    full_path = gguf_file.resolve().as_posix()

    text = f"FROM {full_path}\n"
    text = text + f'TEMPLATE """{TEMPLATE}"""\n'
    text = text + 'PARAMETER stop "<|im_end|>"\n'
    text = text + "PARAMETER temperature 0\n"
    text = text + "PARAMETER num_ctx 4096\n"
    return text


def create_ollama_model(gguf_file, name, ollama_exe=OLLAMA_EXE):
    """Run `ollama create` for one GGUF file."""
    if gguf_file.suffix != ".gguf" or not gguf_file.exists():
        raise UserError(f"{gguf_file} is not an existing .gguf file. See docs/steps/09-gguf-ollama.md.")

    modelfile = paths.MODELS / f"Modelfile.{name}"
    modelfile.parent.mkdir(parents=True, exist_ok=True)
    modelfile.write_text(modelfile_text(gguf_file), encoding="utf-8")

    command = [str(ollama_exe), "create", name, "-f", str(modelfile)]
    log.info("Running: %s", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise UserError(f"ollama create failed (exit code {result.returncode}). See the messages above.")
    log.info("Created Ollama model '%s' from %s", name, gguf_file.name)
