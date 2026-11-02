"""Tests for the pipeline stages. Everything runs on tiny fake data in a temporary folder."""

import json

import pandas as pd
import pytest

from conftest import FakeClient, label_dict, valid_answer
from triage import paths
from triage.data import splits
from triage.data.download import collect_tags, counts_table, crosstab_table, get_count, make_id
from triage.data.merge import merge_hr_labels
from triage.data.mining import get_score, keyword_score, mine_rare_classes, split_into_words
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
    # "payroll" and "sick leave" count. "three" contains "hr" but is not the word "hr".
    assert keyword_score(text.lower(), split_into_words(text), keywords) == 2


def test_get_score_reads_the_score_field():
    assert get_score({"score": 3, "ticket": {}}) == 3


def test_an_unknown_mining_round_is_refused():
    with pytest.raises(UserError, match="Unknown round"):
        mine_rare_classes(9)


# ---- download helpers (the download itself needs the internet, so only the small helpers are tested)


def test_make_id_is_stable_and_twelve_characters_long():
    assert make_id("hello") == make_id("hello")
    assert len(make_id("hello")) == 12
    assert make_id("hello") != make_id("hello!")


def test_counts_table_shows_counts_and_percentages():
    text = counts_table(pd.Series(["a", "a", "b"]), "letters")
    assert "### letters" in text
    assert "| a | 2 | 66.7 |" in text
    assert "| b | 1 | 33.3 |" in text


def test_crosstab_table_has_a_header_and_one_row_per_queue():
    queues = pd.Series(["q1", "q1", "q2"], name="queue")
    priorities = pd.Series(["low", "high", "low"])
    text = crosstab_table(queues, priorities)
    assert text.startswith("| queue | high | low |")
    assert "| q1 | 50.0 | 50.0 |" in text
    assert "| q2 | 0.0 | 100.0 |" in text


def test_collect_tags_keeps_only_real_text():
    assert collect_tags(["x", None, "", float("nan"), "y"]) == ["x", "y"]


def test_get_count_reads_the_second_item_of_a_pair():
    assert get_count(("tag", 7)) == 7


# ---- merge hr into customer_service (ADR-013) --------------------------------------------------


def test_merge_changes_hr_to_customer_service_and_keeps_a_trace(tmp_path):
    labeled = tmp_path / "labeled"
    merged = tmp_path / "merged"
    rows = [
        {"id": "a", "schema_version": "1.0", "teacher": "t", "label": label_dict(category="hr")},
        {"id": "b", "schema_version": "1.0", "teacher": "t", "label": label_dict(category="billing")},
    ]
    write_jsonl(labeled / "val__t.jsonl", rows)

    counts = merge_hr_labels(teacher="t", labeled_dir=labeled, merged_dir=merged)

    # Put the merged rows in a dictionary keyed by id, to look them up easily.
    merged_rows = {}
    for row in read_jsonl(merged / "val.jsonl"):
        merged_rows[row["id"]] = row

    assert merged_rows["a"]["label"]["category"] == "customer_service"
    assert merged_rows["a"]["merged_from"] == "hr"
    assert merged_rows["a"]["schema_version"] == "1.1"
    assert merged_rows["b"]["label"]["category"] == "billing"
    assert "merged_from" not in merged_rows["b"]
    assert counts["val"]["customer_service"] == 1

    # The raw record is untouched.
    raw_rows = read_jsonl(labeled / "val__t.jsonl")
    assert len(raw_rows) == 2
    assert raw_rows[0]["label"]["category"] == "hr"


# ---- training data -----------------------------------------------------------------------------


def test_make_examples_builds_prompt_and_completion(tmp_path):
    write_jsonl(tmp_path / "m.jsonl", [{"id": "a", "label": label_dict(category="outage")}])
    examples = make_examples(tmp_path / "m.jsonl", {"a": "The site is down"})
    assert len(examples) == 1
    example = examples[0]

    roles = []
    for message in example["prompt"]:
        roles.append(message["role"])
    assert roles == ["system", "user"]
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


def make_raw_tickets(n):
    """n fake tickets, spread evenly over four queues."""
    queues = ["Technical Support", "Product Support", "Billing and Payments", "Customer Service"]
    rows = []
    for i in range(n):
        row = {}
        row["id"] = f"id{i}"
        row["text"] = f"unique ticket text number {i} about thing {i * 7}"
        row["queue"] = queues[i % 4]
        row["priority"] = "low"
        row["version"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def ids_of(rows):
    """The set of ticket ids in a list of rows."""
    ids = set()
    for row in rows:
        ids.add(row["id"])
    return ids


def test_stratified_sample_returns_exactly_n_rows_and_keeps_shares():
    df = make_raw_tickets(400)
    df["category"] = df["queue"]
    sample = splits.stratified_sample(df, 40, "category")
    assert len(sample) == 40
    # Four equal groups, so each should have 10 rows: only one distinct count.
    assert sample["category"].value_counts().nunique() == 1


def test_capped_sample_takes_small_groups_whole_and_caps_the_big_ones():
    df = pd.DataFrame({"category": ["a"] * 5 + ["b"] * 100 + ["c"] * 100, "id": range(205)})
    sample, cap = splits.capped_sample(df, 100, "category")
    counts = sample["category"].value_counts()
    assert counts["a"] == 5  # the rare class is kept whole
    assert cap == 47
    assert counts["b"] == 47
    assert counts["c"] == 47


def test_build_splits_makes_separate_splits_and_freezes_the_test_set(tmp_path):
    raw = tmp_path / "raw.jsonl"
    make_raw_tickets(1500).to_json(raw, orient="records", lines=True)
    out_dir = tmp_path / "processed"
    report = tmp_path / "splits.md"

    splits.build_splits(raw_file=raw, out_dir=out_dir, report_file=report)

    test = read_jsonl(out_dir / "test.jsonl")
    val = read_jsonl(out_dir / "val.jsonl")
    train = read_jsonl(out_dir / "train.jsonl")
    assert len(test) == splits.N_TEST
    assert len(val) == splits.N_VAL

    gold_count = 0
    for row in test:
        if row["gold"]:
            gold_count = gold_count + 1
    assert gold_count == splits.N_GOLD

    # No ticket may appear in two splits.
    test_ids = ids_of(test)
    val_ids = ids_of(val)
    train_ids = ids_of(train)
    assert len(test_ids & val_ids) == 0
    assert len(test_ids & train_ids) == 0
    assert len(val_ids & train_ids) == 0
    assert report.exists()

    # A second run must refuse to overwrite the frozen test set.
    with pytest.raises(UserError, match="frozen"):
        splits.build_splits(raw_file=raw, out_dir=out_dir, report_file=report)


# ---- teacher labelling (no network: a fake client answers) -------------------------------------


def local_endpoint():
    return load_endpoint("base_qwen")  # on this computer, so no payment approval is needed


def make_tickets(n):
    """n fake tickets with ids t0, t1, ..."""
    tickets = []
    for i in range(n):
        tickets.append({"id": f"t{i}", "text": f"ticket {i}"})
    return tickets


def test_label_one_retries_once_on_invalid_json_then_succeeds():
    client = FakeClient(["not json", valid_answer()])
    label, details = label_one(client, local_endpoint(), "m", "ticket")
    assert label.category.value == "technical"
    assert client.calls == 2
    assert details["input_tokens"] == 10


def test_label_one_raises_after_two_bad_answers():
    client = FakeClient(["nope"])
    with pytest.raises(ValueError):
        label_one(client, local_endpoint(), "m", "ticket")
    assert client.calls == 2


def test_label_split_is_resumable_and_logs_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "LABELED", tmp_path / "labeled")
    write_jsonl(paths.PROCESSED / "val.jsonl", make_tickets(3))

    # Ticket 1 is fine, ticket 2 is invalid twice (a failure), ticket 3 is fine.
    client = FakeClient([valid_answer(), "bad", "bad", valid_answer()])
    result = label_split("val", model="m", client=client, endpoint=local_endpoint())
    assert result.labelled == 2
    assert result.failed == 1
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
    write_jsonl(paths.PROCESSED / "val.jsonl", make_tickets(1))
    client = FakeClient([valid_answer()])
    with pytest.raises(UserError, match="--yes"):
        label_split("val", client=client, endpoint=load_endpoint("teacher"))
    assert client.calls == 0  # nothing was sent


# ---- eval runner and scoring ---------------------------------------------------------------------


def write_gold_test_file(tmp_path, monkeypatch, n=3):
    """A fake test.jsonl with n gold tickets and one ticket that is not gold."""
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    rows = []
    for i in range(n):
        rows.append({"id": f"g{i}", "text": f"ticket {i}", "gold": True})
    rows.append({"id": "not-gold", "text": "ignore me", "gold": False})
    write_jsonl(paths.PROCESSED / "test.jsonl", rows)


def test_run_model_only_asks_gold_tickets_and_retries_failures(tmp_path, monkeypatch):
    write_gold_test_file(tmp_path, monkeypatch)
    runs = tmp_path / "runs"

    client = FakeClient([valid_answer(), RuntimeError("boom"), valid_answer()])
    run_model("base_qwen", "r", client=client, runs_dir=runs)
    rows = read_jsonl(runs / "r.jsonl")

    ids = []
    for row in rows:
        ids.append(row["id"])
    assert ids == ["g0", "g1", "g2"]  # the ticket that is not gold was never asked
    assert rows[1]["raw"] is None
    assert "boom" in rows[1]["error"]

    client2 = FakeClient([valid_answer()])
    run_model("base_qwen", "r", client=client2, runs_dir=runs)
    assert client2.calls == 1  # only the failed ticket is asked again


def test_score_run_counts_invalid_answers_as_wrong_and_keeps_the_latest_row(tmp_path):
    gold = []
    for i in range(4):
        label = {"id": f"g{i}", "category": "technical", "priority": "high"}
        label["sentiment"] = "negative"
        label["products"] = ["Jira"]
        gold.append(label)

    gold_file = tmp_path / "gold.jsonl"
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    write_jsonl(gold_file, gold)
    write_jsonl(
        runs / "r.jsonl",
        [
            {"id": "g0", "raw": valid_answer(), "latency_s": 1.0},  # right
            {"id": "g1", "raw": valid_answer(category="billing"), "latency_s": 2.0},  # wrong category
            {"id": "g2", "raw": "Sure! Here you go", "latency_s": 3.0},  # not JSON
            {"id": "g3", "raw": None, "error": "boom", "latency_s": 0.0},  # a failed call...
            {"id": "g3", "raw": valid_answer(), "latency_s": 4.0},  # ...then the retry
        ],
    )

    summary, lines = score_run("r", runs_dir=runs, reports_dir=reports, gold_file=gold_file)

    assert summary["tickets"] == 4  # g3 is counted once, from its latest row
    assert summary["json_valid"] == 0.75
    assert summary["schema_valid"] == 0.75
    assert summary["category_accuracy"] == 0.5  # g0 and g3 right; g1 wrong; g2 invalid
    assert summary["products_f1"] == 0.75  # three answered with Jira, the invalid one scores 0
    assert (reports / "r.md").exists()
    assert (reports / "r.json").exists()

    found_headline = False
    for line in lines:
        if "category accuracy" in line:
            found_headline = True
    assert found_headline
