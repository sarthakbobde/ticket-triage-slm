"""Merge the trained LoRA adapter into the base model (Step 09).

Training saved only a small adapter (about 120 MB): the changes on top of the frozen base model.
To serve the model with Ollama or llama.cpp we need ONE ordinary model, so we add the adapter's
changes into the base weights ("merge") and save the result.

This needs the optional `export` packages (torch, transformers, peft):
    uv sync --extra serve --extra export

Memory: the 3B model is about 6 GB in 16-bit. Close other programs first. The merge runs on the CPU
because the 4 GB graphics card is too small, and takes a few minutes.
"""

import logging

from triage.errors import UserError

log = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def merge_adapter(adapter_dir, out_dir, base_model=BASE_MODEL):
    """Merge the adapter in `adapter_dir` into `base_model` and save the full model to `out_dir`."""
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise UserError(f"No adapter_model.safetensors in {adapter_dir}. Unzip the Kaggle adapter there first.")

    # These packages are big and optional, so we only import them when this function really runs.
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise UserError("Merging needs torch, transformers and peft: uv sync --extra serve --extra export") from error

    # Load the base model in its own number format (bfloat16), so no precision is lost.
    log.info("Loading base model %s (about 6 GB, on the CPU)...", base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    log.info("Applying the adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()  # adds the adapter's changes into the base weights

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving the merged model to %s", out_dir)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return out_dir
