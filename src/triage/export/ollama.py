"""Import the merged model into Ollama at a chosen compression level (Step 09).

Ollama reads the merged Hugging Face folder, converts it to its GGUF format, and can shrink
("quantize") the numbers on the way in. Smaller numbers mean a smaller, faster model, but
a little less accurate. Step 09 measures how much quality each level costs.

    q8_0     8 bits per number: close to the original, about 3.3 GB
    q5_K_M   5 bits:  about 2.2 GB
    q4_K_M   4 bits:  about 1.9 GB (what the plain Ollama qwen2.5:3b model uses)
    f16      no shrinking (reference): about 6 GB, too big for the 4 GB card
"""

import logging
import subprocess
from pathlib import Path

from triage import paths
from triage.errors import UserError

log = logging.getLogger(__name__)

QUANTS = ["q8_0", "q5_K_M", "q4_K_M"]
OLLAMA_EXE = Path("E:/Ollama/ollama.exe")

# The chat format Qwen was trained with (ChatML). The training notebook built its examples with the
# same format from the tokenizer, so serving must use it too: our messages are always system + user.
TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""


def modelfile_text(model_dir: Path) -> str:
    """The Ollama recipe for the model: where the weights are, the chat format, and when to stop."""
    return (
        f"FROM {model_dir.as_posix()}\n"
        f'TEMPLATE """{TEMPLATE}"""\n'
        'PARAMETER stop "<|im_end|>"\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_ctx 4096\n"
    )


def create_ollama_model(model_dir: Path, name: str, quant: str, ollama_exe: Path = OLLAMA_EXE) -> None:
    """Run `ollama create`, which converts the folder and quantizes it."""
    if quant != "f16" and quant not in QUANTS:
        raise UserError(f"Unknown quantization '{quant}'. Choose from {QUANTS} or f16.")
    if not (model_dir / "config.json").exists():
        raise UserError(
            f"{model_dir} is not a merged model folder (no config.json). Run `triage export merge` first."
        )

    modelfile = paths.ROOT / "models" / f"Modelfile.{name}"
    modelfile.parent.mkdir(parents=True, exist_ok=True)
    modelfile.write_text(modelfile_text(model_dir), encoding="utf-8")

    command = [str(ollama_exe), "create", name, "-f", str(modelfile)]
    if quant != "f16":
        command += ["--quantize", quant]
    log.info("Running: %s", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise UserError(f"ollama create failed (exit code {result.returncode}). See the messages above.")
    log.info("Created Ollama model '%s' (%s)", name, quant)
