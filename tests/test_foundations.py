"""Tests for the small shared pieces: text cleaning, jsonl helpers, config, schema, CLI parsing."""

import pytest
import yaml

from triage import paths
from triage.cli import build_parser, cmd_eval_run, main
from triage.dataset_map import QUEUE_TO_CATEGORY
from triage.errors import UserError
from triage.io import append_row, read_done_ids, read_jsonl, write_jsonl
from triage.llm import is_local_url, require_approval_for_paid_calls, response_format_for
from triage.prompts import SYSTEM_PROMPT
from triage.schema import SCHEMA_VERSION, Category
from triage.settings import ModelEndpoint, load_endpoint
from triage.textclean import clean_text, normalise

# ---- text cleaning ---------------------------------------------------------------------------


def test_clean_text_turns_escapes_and_html_into_whitespace():
    assert clean_text("a\\nb<br>c\\td") == "a\nb\nc d"


def test_normalise_ignores_case_punctuation_and_spacing():
    assert normalise("Hello,   WORLD!") == normalise("hello world")


# ---- jsonl helpers -----------------------------------------------------------------------------


def test_write_then_read_roundtrip_keeps_unicode(tmp_path):
    rows = [{"id": "a", "text": "café"}, {"id": "b", "text": "x"}]
    write_jsonl(tmp_path / "sub" / "f.jsonl", rows)  # also creates the folder
    assert read_jsonl(tmp_path / "sub" / "f.jsonl") == rows


def test_read_done_ids_can_ignore_failed_rows(tmp_path):
    path = tmp_path / "run.jsonl"
    with open(path, "w", encoding="utf-8") as file:
        append_row(file, {"id": "ok", "raw": "x"})
        append_row(file, {"id": "bad", "raw": None, "error": "boom"})
    assert read_done_ids(path) == {"ok", "bad"}
    assert read_done_ids(path, ignore_failed=True) == {"ok"}
    assert read_done_ids(tmp_path / "missing.jsonl") == set()


# ---- config ------------------------------------------------------------------------------------


def test_every_section_of_the_real_models_yaml_loads():
    sections = yaml.safe_load(paths.MODELS_CONFIG.read_text(encoding="utf-8"))
    for name in sections:
        assert isinstance(load_endpoint(name), ModelEndpoint)


def test_config_typo_is_an_error_not_silently_ignored(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("teacher:\n  base_ulr: http://x\n", encoding="utf-8")
    with pytest.raises(UserError):
        load_endpoint("teacher", config)


def test_unknown_config_section_lists_the_available_ones(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("teacher:\n  base_url: http://x\n", encoding="utf-8")
    with pytest.raises(UserError, match="teacher"):
        load_endpoint("nope", config)


# ---- model calls: format option and cost guard -------------------------------------------------


def test_response_format_options():
    assert response_format_for("none") is None
    assert response_format_for("json_object") == {"type": "json_object"}
    assert response_format_for("json_schema")["json_schema"]["name"] == "TicketTriage"


def test_remote_endpoints_need_approval_but_local_ones_do_not():
    remote = ModelEndpoint(base_url="https://example.com/v1/")
    local = ModelEndpoint(base_url="http://localhost:11434/v1")
    assert is_local_url(local.base_url) and not is_local_url(remote.base_url)
    require_approval_for_paid_calls(local, 100, approved=False)  # fine
    require_approval_for_paid_calls(remote, 100, approved=True)  # fine
    with pytest.raises(UserError, match="--yes"):
        require_approval_for_paid_calls(remote, 100, approved=False)


# ---- the contract: schema, prompt, dataset mapping ---------------------------------------------


def test_schema_has_the_five_categories_and_no_hr():
    assert {c.value for c in Category} == {"technical", "product", "billing", "outage", "customer_service"}
    assert SCHEMA_VERSION == "1.1"


def test_prompt_lists_every_category():
    for category in Category:
        assert category.value in SYSTEM_PROMPT


def test_every_dataset_queue_maps_to_a_known_category():
    for queue, category in QUEUE_TO_CATEGORY.items():
        assert isinstance(category, Category), queue
    assert QUEUE_TO_CATEGORY["Human Resources"] is Category.CUSTOMER_SERVICE  # ADR-013


# ---- command line ------------------------------------------------------------------------------


def test_eval_run_option_does_not_clobber_the_command_handler():
    # Regression: the handler used to be stored under the name "run", the same as --run.
    args = build_parser().parse_args(["eval", "run", "--config", "base_qwen", "--run", "my-run", "--yes"])
    assert args.run == "my-run"
    assert args.handler is cmd_eval_run
    assert args.yes is True


def test_user_mistakes_print_a_message_and_exit_2():
    assert main(["label"]) == 2  # --split is required


def test_mining_round_must_be_known():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["data", "mine", "--round", "3"])
