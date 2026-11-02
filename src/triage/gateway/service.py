"""The gateway's decision logic (Step 11): ask the small model first, hand hard tickets to the teacher.

For each ticket:
  1. Ask the student (the small model on this laptop) and read how sure it was of each decision.
  2. Use the student's answer if it is valid JSON that fits the schema AND it is confident enough.
  3. Otherwise ask the teacher (a big paid model), if one is configured and its budget is not spent.

The teacher costs money, so it has a hard cap on the number of calls (`teacher_budget`).
"""

import logging
import time
import uuid
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from triage.errors import UserError
from triage.gateway.confidence import field_confidences, overall_confidence
from triage.gateway.metrics import GatewayMetrics
from triage.gateway.schemas import TriageResponse
from triage.labeling.teacher import label_one
from triage.llm import make_client, require_approval_for_paid_calls
from triage.prompts import build_messages
from triage.schema import TicketTriage
from triage.settings import ModelEndpoint, load_endpoint

log = logging.getLogger(__name__)


class GatewayError(Exception):
    """No usable answer could be produced for this ticket."""


class StudentAnswer:
    """What came back from the student for one ticket."""

    def __init__(self, result: TicketTriage | None, confidence: float | None, problem: str | None):
        self.result = result  # None if the answer was unusable
        self.confidence = confidence
        self.problem = problem  # None, "invalid_output" or "student_error"


def ask_student(client: OpenAI, model: str, text: str) -> StudentAnswer:
    """One call to the small model. Never raises: problems are reported in the result."""
    try:
        reply = client.chat.completions.create(
            model=model, messages=build_messages(text), temperature=0, logprobs=True, max_tokens=700
        )
    except Exception as error:  # noqa: BLE001 - a down or slow student must trigger the fallback
        log.warning("student call failed: %s", str(error)[:200])
        return StudentAnswer(None, None, "student_error")

    choice = reply.choices[0]
    content = choice.message.content or ""
    try:
        result = TicketTriage.model_validate_json(content)
    except ValidationError:
        return StudentAnswer(None, None, "invalid_output")

    tokens = []
    if choice.logprobs is not None and choice.logprobs.content:
        for item in choice.logprobs.content:
            tokens.append((item.token, item.logprob))
    return StudentAnswer(result, overall_confidence(field_confidences(content, tokens)), None)


class TriageService:
    def __init__(
        self,
        student_client: OpenAI,
        student_model: str,
        metrics: GatewayMetrics,
        min_confidence: float = 0.0,
        teacher_client: OpenAI | None = None,
        teacher_endpoint: ModelEndpoint | None = None,
        teacher_model: str | None = None,
        teacher_budget: int = 0,
    ):
        self.student_client = student_client
        self.student_model = student_model
        self.metrics = metrics
        self.min_confidence = min_confidence  # answers less sure than this go to the teacher
        self.teacher_client = teacher_client
        self.teacher_endpoint = teacher_endpoint
        self.teacher_model = teacher_model
        self.teacher_budget = teacher_budget  # remaining paid calls; 0 means "never call the teacher"

    def can_use_teacher(self) -> bool:
        return self.teacher_client is not None and self.teacher_budget > 0

    def triage(self, text: str, request_id: str | None = None) -> TriageResponse:
        request_id = request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()

        answer = ask_student(self.student_client, self.student_model, text)
        reason = answer.problem
        if reason is None and answer.confidence is not None and answer.confidence < self.min_confidence:
            reason = "low_confidence"

        if reason is None and answer.result is not None:
            return self._finish(request_id, "student", answer.result, answer.confidence, None, started)

        # The student's answer is unusable or not trusted. Try the teacher.
        if self.can_use_teacher():
            teacher_result = self._ask_teacher(text)
            if teacher_result is not None:
                return self._finish(request_id, "teacher", teacher_result, answer.confidence, reason, started)

        # No teacher available. A valid-but-unsure student answer is still better than nothing.
        if answer.result is not None:
            log.info("id=%s low confidence but no teacher available: answering with the student", request_id)
            return self._finish(request_id, "student", answer.result, answer.confidence, None, started)

        self.metrics.record_failure()
        raise GatewayError(f"No usable answer ({reason}); the teacher is not available.")

    def _ask_teacher(self, text: str) -> TicketTriage | None:
        self.teacher_budget -= 1  # count the attempt before it is made: a failed call may still cost money
        try:
            result, _details = label_one(self.teacher_client, self.teacher_endpoint, self.teacher_model, text)
            return result
        except Exception as error:  # noqa: BLE001
            log.warning("teacher call failed: %s", str(error)[:200])
            return None

    def _finish(
        self,
        request_id: str,
        source: str,
        result: TicketTriage,
        confidence: float | None,
        fallback_reason: str | None,
        started: float,
    ) -> TriageResponse:
        latency_ms = int((time.perf_counter() - started) * 1000)
        self.metrics.record_answer(source, latency_ms, fallback_reason)
        log.info(
            "id=%s source=%s reason=%s confidence=%s latency_ms=%d",
            request_id,
            source,
            fallback_reason,
            None if confidence is None else round(confidence, 3),
            latency_ms,
        )
        return TriageResponse(
            request_id=request_id,
            source=source,
            result=result,
            confidence=confidence,
            fallback_reason=fallback_reason,
            latency_ms=latency_ms,
        )


def build_service(
    student_config: str = "student_q4",
    use_teacher: bool = False,
    approved: bool = False,
    teacher_budget: int = 0,
    min_confidence: float = 0.0,
    teacher_cost_per_ticket: float = 0.002,
) -> TriageService:
    """Wire up the real models from configs/models.yaml."""
    student = load_endpoint(student_config)
    options: dict[str, Any] = {}
    if use_teacher:
        teacher = load_endpoint("teacher")
        if teacher_budget <= 0:
            raise UserError("--teacher needs a positive --teacher-budget (the most paid calls allowed).")
        require_approval_for_paid_calls(teacher, teacher_budget, approved)
        options = {
            "teacher_client": make_client(teacher, max_retries=2, timeout=60),
            "teacher_endpoint": teacher,
            "teacher_model": teacher.model,
            "teacher_budget": teacher_budget,
        }
    return TriageService(
        student_client=make_client(student, max_retries=0, timeout=60),
        student_model=student.model or "",
        metrics=GatewayMetrics(teacher_cost_per_ticket),
        min_confidence=min_confidence,
        **options,
    )
