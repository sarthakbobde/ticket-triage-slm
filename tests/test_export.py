"""Tests for the Step 09 export code that does not need torch or Ollama."""

import pytest

from triage.cli import build_parser
from triage.errors import UserError
from triage.export.merge import merge_adapter
from triage.export.ollama import create_ollama_model, modelfile_text


def test_modelfile_points_at_the_folder_and_uses_the_chatml_format(tmp_path):
    text = modelfile_text(tmp_path / "merged")
    assert text.startswith("FROM ")
    assert "<|im_start|>system" in text and "{{ .Prompt }}" in text  # system + user, as in training
    assert 'PARAMETER stop "<|im_end|>"' in text
    assert "PARAMETER temperature 0" in text


def test_import_refuses_a_missing_or_wrong_file(tmp_path):
    with pytest.raises(UserError, match="existing .gguf file"):
        create_ollama_model(tmp_path / "nope.gguf", "x")
    (tmp_path / "model.bin").write_text("x")
    with pytest.raises(UserError, match="existing .gguf file"):
        create_ollama_model(tmp_path / "model.bin", "x")


def test_merge_refuses_a_folder_without_an_adapter(tmp_path):
    with pytest.raises(UserError, match="adapter_model.safetensors"):
        merge_adapter(tmp_path, tmp_path / "out")


def test_export_commands_parse():
    args = build_parser().parse_args(["export", "ollama", "--gguf", "a.gguf", "--name", "t-q4"])
    assert args.name == "t-q4" and args.gguf == "a.gguf"
    assert build_parser().parse_args(["export", "merge"]).adapter.endswith("qwen-triage-lora-v1")
