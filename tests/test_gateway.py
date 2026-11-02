"""Tests for the gateway (Step 11). Fake models only: no network, no GPU, no paid API."""

import pytest
from fastapi.testclient import TestClient

from conftest import FakeClient, tokens_for, valid_answer
from triage.cli import build_parser
from triage.errors import UserError
from triage.gateway.app import create_app
from triage.gateway.confidence import field_confidences, overall_confidence
from triage.gateway.metrics import GatewayMetrics
from triage.gateway.service import GatewayError, TriageService, build_service
from triage.settings import ModelEndpoint

ANSWER = valid_answer(category="technical", priority="high", sentiment="negative")
TEACHER_ANSWER = valid_answer(category="billing", priority="low", sentiment="neutral")
SURE = (ANSWER, tokens_for(ANSWER))
UNSURE = (ANSWER, tokens_for(ANSWER, {"priority": 0.5}))  # the student hesitated on priority


def make_service(student_answers, teacher_answers=None, min_confidence=0.0, budget=5):
    # No teacher answers given means: this service has no teacher at all.
    teacher = None
    teacher_budget = 0
    if teacher_answers is not None:
        teacher = FakeClient(teacher_answers)
        teacher_budget = budget

    service = TriageService(
        student_client=FakeClient(student_answers),
        student_model="student",
        metrics=GatewayMetrics(teacher_cost_per_ticket=0.002),
        min_confidence=min_confidence,
        teacher_client=teacher,
        teacher_endpoint=ModelEndpoint(base_url="http://localhost:1/v1"),
        teacher_model="teacher",
        teacher_budget=teacher_budget,
    )
    return service, teacher


# ---- confidence --------------------------------------------------------------------------------


def test_confidence_is_the_weakest_of_the_three_decisions():
    tokens = tokens_for(ANSWER, {"category": 0.9, "priority": 0.6, "sentiment": 0.95})
    confidences = field_confidences(ANSWER, tokens)
    assert confidences["category"] == pytest.approx(0.9)
    assert confidences["priority"] == pytest.approx(0.6)
    assert overall_confidence(confidences) == pytest.approx(0.6)


def test_confidence_is_unknown_when_tokens_do_not_match_the_text():
    assert field_confidences(ANSWER, [("something else", 0.0)]) == {}
    assert overall_confidence({}) is None


# ---- the decision logic ------------------------------------------------------------------------


def test_a_confident_student_answers_without_the_teacher():
    service, teacher = make_service([SURE], [TEACHER_ANSWER])
    response = service.triage("ticket")
    assert response.source == "student" and response.fallback_reason is None
    assert response.confidence == pytest.approx(1.0)
    assert teacher.calls == 0


def test_invalid_student_output_falls_back_to_the_teacher():
    service, teacher = make_service(["Sure! Here is the JSON"], [TEACHER_ANSWER])
    response = service.triage("ticket")
    assert response.source == "teacher" and response.fallback_reason == "invalid_output"
    assert response.result.category.value == "billing"
    assert service.teacher_budget == 4


def test_a_student_error_falls_back_to_the_teacher():
    service, _ = make_service([RuntimeError("connection refused")], [TEACHER_ANSWER])
    assert service.triage("ticket").fallback_reason == "student_error"


def test_an_unsure_student_falls_back_to_the_teacher():
    service, _ = make_service([UNSURE], [TEACHER_ANSWER], min_confidence=0.8)
    response = service.triage("ticket")
    assert response.source == "teacher" and response.fallback_reason == "low_confidence"
    assert response.confidence == pytest.approx(0.5)  # we still report how unsure the student was


def test_an_unsure_student_is_used_anyway_when_no_teacher_is_configured():
    service, _ = make_service([UNSURE], teacher_answers=None, min_confidence=0.8)
    response = service.triage("ticket")
    assert response.source == "student" and response.result.category.value == "technical"


def test_nothing_usable_raises_and_is_counted():
    service, _ = make_service(["not json"], teacher_answers=None)
    with pytest.raises(GatewayError):
        service.triage("ticket")
    assert service.metrics.snapshot()["requests_failed"] == 1


def test_the_teacher_budget_is_a_hard_cap():
    service, teacher = make_service([UNSURE], [TEACHER_ANSWER], min_confidence=0.8, budget=1)
    assert service.triage("one").source == "teacher"  # uses the whole budget
    assert service.triage("two").source == "student"  # budget gone: the unsure student answers
    assert teacher.calls == 1 and not service.can_use_teacher()


def test_a_failed_teacher_call_still_uses_budget_and_falls_back_to_the_student():
    service, teacher = make_service([UNSURE], [RuntimeError("quota")], min_confidence=0.8, budget=2)
    assert service.triage("ticket").source == "student"
    assert teacher.calls == 1 and service.teacher_budget == 1


def test_metrics_count_answers_fallbacks_and_savings():
    service, _ = make_service([SURE, SURE, "garbage"], [TEACHER_ANSWER])
    for _ in range(3):
        service.triage("ticket")
    snapshot = service.metrics.snapshot()
    assert snapshot["answered_by_student"] == 2 and snapshot["answered_by_teacher"] == 1
    assert snapshot["fallback_rate"] == pytest.approx(1 / 3)
    assert snapshot["fallback_reasons"] == {"invalid_output": 1}
    assert snapshot["estimated_cost_saved_usd"] == pytest.approx(0.004)


# ---- the HTTP layer ----------------------------------------------------------------------------


def test_post_triage_returns_the_answer_and_echoes_the_request_id():
    service, _ = make_service([SURE])
    client = TestClient(create_app(service))
    response = client.post("/triage", json={"text": "My login fails"}, headers={"X-Request-ID": "abc123"})
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "abc123" and body["source"] == "student"
    assert body["result"]["category"] == "technical" and body["fallback_reason"] is None


def test_empty_or_missing_text_is_rejected_with_422():
    client = TestClient(create_app(make_service([SURE])[0]))
    assert client.post("/triage", json={"text": ""}).status_code == 422
    assert client.post("/triage", json={}).status_code == 422


def test_an_unusable_answer_is_a_503_not_a_crash():
    client = TestClient(create_app(make_service(["garbage"])[0]))
    response = client.post("/triage", json={"text": "ticket"})
    assert response.status_code == 503 and "No usable answer" in response.json()["detail"]


def test_health_and_metrics_endpoints():
    service, _ = make_service([SURE], [TEACHER_ANSWER], budget=3)
    client = TestClient(create_app(service))
    assert client.get("/health").json() == {
        "status": "ok",
        "student_model": "student",
        "teacher_available": True,
    }
    client.post("/triage", json={"text": "ticket"})
    metrics = client.get("/metrics").json()
    assert metrics["requests_answered"] == 1 and metrics["teacher_budget_left"] == 3


# ---- safety around paid calls ------------------------------------------------------------------


def test_the_teacher_needs_a_budget_and_an_explicit_yes():
    with pytest.raises(UserError, match="teacher-budget"):
        build_service(use_teacher=True, teacher_budget=0)
    with pytest.raises(UserError, match="--yes"):
        build_service(use_teacher=True, teacher_budget=10, approved=False)


def test_serve_command_defaults_are_safe():
    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1" and args.teacher is False and args.teacher_budget == 0


# ---- threshold simulation ------------------------------------------------------------------------


def test_simulation_moves_unsure_tickets_to_the_teacher():
    from triage.gateway.calibrate import simulate

    gold = {
        "a": {"category": "billing", "priority": "low", "sentiment": "neutral"},
        "b": {"category": "technical", "priority": "high", "sentiment": "negative"},
    }
    # The teacher is always right: its answers are copies of the gold labels.
    teacher = {}
    for ticket_id in gold:
        teacher[ticket_id] = dict(gold[ticket_id])
    student = {
        "a": {
            "answer": {"category": "billing", "priority": "low", "sentiment": "neutral"},
            "confidence": 0.99,
        },
        # unsure AND wrong: the teacher should rescue this one when the threshold is high enough
        "b": {
            "answer": {"category": "product", "priority": "high", "sentiment": "negative"},
            "confidence": 0.6,
        },
    }
    # Put the rows in a dictionary keyed by threshold, so we can look each one up.
    rows = {}
    for row in simulate(student, teacher, gold):
        rows[row["threshold"]] = row
    assert rows[0.0]["to_teacher"] == 0 and rows[0.0]["category"] == 0.5  # student only: one wrong
    assert rows[0.7]["to_teacher"] == 0.5 and rows[0.7]["category"] == 1.0  # ticket b went to the teacher
    assert rows[0.99]["to_teacher"] == 0.5  # ticket a (confidence 0.99) is not below 0.99, so it stays


def test_an_unusable_student_answer_always_goes_to_the_teacher_in_the_simulation():
    from triage.gateway.calibrate import simulate

    gold = {"a": {"category": "billing", "priority": "low", "sentiment": "neutral"}}
    teacher = {"a": dict(gold["a"])}
    student = {"a": {"answer": None, "confidence": None}}
    assert simulate(student, teacher, gold)[0]["to_teacher"] == 1.0  # even at threshold 0


def test_eval_calibrate_command_parses():
    args = build_parser().parse_args(["eval", "calibrate", "--analyse-only"])
    assert args.analyse_only and args.run == "gateway-student-q4"
