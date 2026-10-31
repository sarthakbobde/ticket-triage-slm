"""Tests for the pipeline stages. Everything runs on tiny fake data in a temp folder."""

import json

import pandas as pd
import pytest

from conftest import FakeClient, label_dict, valid_answer
from triage import paths
from triage.data import splits
from triage.data.merge import merge_hr_labels
from triage.data.mining import get_score, keyword_score, split_into_words
from triage.data.training import check_no_leakage, make_examples
from triage.errors import UserError
from triage.evals.runner import run_model
from triage.evals.scoring import score_run
from triage.io import read_jsonl, write_jsonl
from triage.labeling.teacher import label_one, label_split
from triage.settings import load_endpoint

# ---- mining ------------------------------------------------------------------------------------


def test_split_into_words_lowercases_and_drops_punctuation():
    assert split_into_words("Hello, World! HR-team") == ["hello", "world", "hr", "team"]


def test_keyword_score_matches_whole_words_and_phrases():
    keywords = {"words": ["hr", "payroll"], "phrases": ["sick leave"]}
    text = "Question about payroll and sick leave for the three of us"
    assert keyword_score(text.lower(), split_into_words(text), keywords) == 2  # "three" is not "hr"


def test_get_score_reads_the_score_field():
    assert get_score({"score": 3, "ticket": {}}) == 3


# ---- merge hr into customer_service (ADR-013) --------------------------------------------------


def test_merge_changes_hr_to_customer_service_and_keeps_a_trace(tmp_path):
    labeled, merged = tmp_path / "labeled", tmp_path / "merged"
    rows = [
        {"id": "a", "schema_version": "1.0", "teacher": "t", "label": label_dict(category="hr")},
        {"id": "b", "schema_version": "1.0", "teacher": "t", "label": label_dict(category="billing")},
    ]
    write_jsonl(labeled / "val__t.jsonl", rows)

    counts = merge_hr_labels(teacher="t", labeled_dir=labeled, merged_dir=merged)

    out = {row["id"]: row for row in read_jsonl(merged / "val.jsonl")}
    assert out["a"]["label"]["category"] == "customer_service"
    assert out["a"]["merged_from"] == "hr"
    assert out["b"]["label"]["category"] == "billing" and "merged_from" not in out["b"]
    assert out["a"]["schema_version"] == "1.1"
    assert counts["val"]["customer_service"] == 1
    assert len(read_jsonl(labeled / "val__t.jsonl")) == 2  # the raw record is untouched
    assert read_jsonl(labeled / "val__t.jsonl")[0]["label"]["category"] == "hr"


# ---- training data -----------------------------------------------------------------------------


def test_make_examples_builds_prompt_and_completion(tmp_path):
    write_jsonl(tmp_path / "m.jsonl", [{"id": "a", "label": label_dict(category="outage")}])
    [example] = make_examples(tmp_path / "m.jsonl", {"a": "The site is down"})
    assert [m["role"] for m in example["prompt"]] == ["system", "user"]
    assert "The site is down" in example["prompt"][1]["content"]
    answer = json.loads(example["completion"][0]["content"])
    assert answer["category"] == "outage"
    assert list(answer) == ["category", "priority", "sentiment", "products", "summary", "suggested_reply"]


def test_leakage_check_rejects_a_test_ticket_in_training():
    train = [{"id": "x"}]
    texts = {"x": "same text"}
    test_rows = [{"id": "x", "text": "same text"}]
    with pytest.raises(AssertionError):
        check_no_leakage(train, [], test_rows, texts)


def test_leakage_check_rejects_a_copy_of_a_test_text():
    train = [{"id": "x"}]
    texts = {"x": "Printer is BROKEN!"}
    test_rows = [{"id": "other", "text": "printer is broken"}]
    with pytest.raises(AssertionError):
        check_no_leakage(train, [], test_rows, texts)


# ---- splits ------------------------------------------------------------------------------------


def make_raw_tickets(n: int) -> pd.DataFrame:
    queues = ["Technical Support", "Product Support", "Billing and Payments", "Customer Service"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"id{i}",
                "text": f"unique ticket text number {i} about thing {i * 7}",
                "queue": queues[i % 4],
                "priority": "low",
                "version": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_stratified_sample_returns_exactly_n_rows_and_keeps_shares():
    df = make_raw_tickets(400)
    df["category"] = df["queue"]
    sample = splits.stratified_sample(df, 40, "category")
    assert len(sample) == 40
    assert sample["category"].value_counts().nunique() == 1  # 4 equal groups -> 10 each


def test_capped_sample_takes_small_groups_whole_and_caps_the_big_ones():
    df = pd.DataFrame({"category": ["a"] * 5 + ["b"] * 100 + ["c"] * 100, "id": range(205)})
    sample, cap = splits.capped_sample(df, 100, "category")
    counts = sample["category"].value_counts()
    assert counts["a"] == 5  # the rare class is kept whole
    assert counts["b"] == counts["c"] == cap == 47


def test_build_splits_makes_disjoint_splits_and_freezes_the_test_set(tmp_path):
    raw = tmp_path / "raw.jsonl"
    make_raw_tickets(1500).to_json(raw, orient="records", lines=True)
    out_dir, report = tmp_path / "processed", tmp_path / "splits.md"

    splits.build_splits(raw_file=raw, out_dir=out_dir, report_file=report)

    test, val, train = (read_jsonl(out_dir / f"{n}.jsonl") for n in ["test", "val", "train"])
    assert (len(test), len(val)) == (splits.N_TEST, splits.N_VAL)
    assert sum(1 for row in test if row["gold"]) == splits.N_GOLD
    ids = [{r["id"] for r in part} for part in (test, val, train)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert report.exists()

    with pytest.raises(UserError, match="frozen"):
        splits.build_splits(raw_file=raw, out_dir=out_dir, report_file=report)


# ---- teacher labelling (no network: a fake client answers) -------------------------------------


def local_endpoint():
    return load_endpoint("base_qwen")  # localhost, so no payment approval is needed


def test_label_one_retries_once_on_invalid_json_then_succeeds():
    client = FakeClient(["not json", valid_answer()])
    label, details = label_one(client, local_endpoint(), "m", "ticket")
    assert label.category.value == "technical" and client.calls == 2
    assert details["input_tokens"] == 10


def test_label_one_raises_after_two_bad_answers():
    client = FakeClient(["nope"])
    with pytest.raises(ValueError):
        label_one(client, local_endpoint(), "m", "ticket")
    assert client.calls == 2


def test_label_split_is_resumable_and_logs_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "LABELED", tmp_path / "labeled")
    write_jsonl(paths.PROCESSED / "val.jsonl", [{"id": f"t{i}", "text": f"ticket {i}"} for i in range(3)])

    # ticket 1 fine, ticket 2 invalid twice (a failure), ticket 3 fine
    client = FakeClient([valid_answer(), "bad", "bad", valid_answer()])
    result = label_split("val", model="m", client=client, endpoint=local_endpoint())
    assert (result.labelled, result.failed) == (2, 1)
    assert len(read_jsonl(paths.LABELED / "val__m.jsonl")) == 2
    assert len(read_jsonl(paths.LABELED / "failures__m.jsonl")) == 1

    # Running again asks only for the one ticket that failed.
    client2 = FakeClient([valid_answer()])
    label_split("val", model="m", client=client2, endpoint=local_endpoint())
    assert client2.calls == 1
    assert len(read_jsonl(paths.LABELED / "val__m.jsonl")) == 3


def test_labelling_with_a_remote_endpoint_is_blocked_without_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "LABELED", tmp_path / "labeled")
    write_jsonl(paths.PROCESSED / "val.jsonl", [{"id": "t0", "text": "ticket"}])
    client = FakeClient([valid_answer()])
    with pytest.raises(UserError, match="--yes"):
        label_split("val", client=client, endpoint=load_endpoint("teacher"))
    assert client.calls == 0  # nothing was sent


# ---- eval runner and scoring ---------------------------------------------------------------------


def write_gold_test_file(tmp_path, monkeypatch, n=3):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    rows = [{"id": f"g{i}", "text": f"ticket {i}", "gold": True} for i in range(n)]
    rows.append({"id": "not-gold", "text": "ignore me", "gold": False})
    write_jsonl(paths.PROCESSED / "test.jsonl", rows)


def test_run_model_only_asks_gold_tickets_and_retries_failures(tmp_path, monkeypatch):
    write_gold_test_file(tmp_path, monkeypatch)
    runs = tmp_path / "runs"

    client = FakeClient([valid_answer(), RuntimeError("boom"), valid_answer()])
    run_model("base_qwen", "r", client=client, runs_dir=runs)
    rows = read_jsonl(runs / "r.jsonl")
    assert [r["id"] for r in rows] == ["g0", "g1", "g2"]  # the non-gold ticket was never asked
    assert rows[1]["raw"] is None and "boom" in rows[1]["error"]

    client2 = FakeClient([valid_answer()])
    run_model("base_qwen", "r", client=client2, runs_dir=runs)
    assert client2.calls == 1  # only the failed ticket is asked again


def test_score_run_counts_invalid_answers_as_wrong_and_keeps_the_latest_row(tmp_path):
    gold = [
        {
            "id": f"g{i}",
            "category": "technical",
            "priority": "high",
            "sentiment": "negative",
            "products": ["Jira"],
        }
        for i in range(4)
    ]
    gold_file, runs, reports = tmp_path / "gold.jsonl", tmp_path / "runs", tmp_path / "reports"
    write_jsonl(gold_file, gold)
    write_jsonl(
        runs / "r.jsonl",
        [
            {"id": "g0", "raw": valid_answer(), "latency_s": 1.0},  # right
            {"id": "g1", "raw": valid_answer(category="billing"), "latency_s": 2.0},  # wrong category
            {"id": "g2", "raw": "Sure! Here you go", "latency_s": 3.0},  # not JSON
            {"id": "g3", "raw": None, "error": "boom", "latency_s": 0.0},  # failed call...
            {"id": "g3", "raw": valid_answer(), "latency_s": 4.0},  # ...then retried
        ],
    )

    summary, lines = score_run("r", runs_dir=runs, reports_dir=reports, gold_file=gold_file)

    assert summary["tickets"] == 4  # g3 counted once, from its latest row
    assert summary["json_valid"] == 0.75 and summary["schema_valid"] == 0.75
    assert summary["category_accuracy"] == 0.5  # g0 and g3 right; g1 wrong; g2 invalid
    assert summary["products_f1"] == 0.75  # three answered with Jira, one invalid scores 0
    assert (reports / "r.md").exists() and (reports / "r.json").exists()
    assert any("category accuracy" in line for line in lines)
