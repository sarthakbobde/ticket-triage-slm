"""The gateway's decision logic (Step 11): ask the small model first, give hard tickets to the teacher.

For every ticket:
  1. Ask the student (the small model running on this laptop) and read how sure it was.
  2. Use the student's answer if it is valid AND the student was sure enough.
  3. Otherwise ask the teacher (a big model that costs money), if one is set up and its budget is left.

The teacher costs money, so it has a hard limit on the number of calls (`teacher_budget`).
"""

import logging
import time
import uuid

from pydantic import ValidationError

from triage.errors import UserError
from triage.gateway.confidence import field_confidences, overall_confidence
from triage.gateway.metrics import GatewayMetrics
from triage.gateway.schemas import TriageResponse
from triage.labeling.teacher import label_one
from triage.llm import make_client, require_approval_for_paid_calls
from triage.prompts import build_messages
from triage.schema import TicketTriage
from triage.settings import load_endpoint

log = logging.getLogger(__name__)


class GatewayError(Exception):
    """Raised when no usable answer could be produced for a ticket."""


class StudentAnswer:
    """What the student gave back for one ticket."""

    def __init__(self, result, confidence, problem):
        self.result = result  # the triage answer, or None if the answer was unusable
        self.confidence = confidence  # how sure the student was (0 to 1), or None if unknown
        self.problem = problem  # None if all is fine, else "invalid_output" or "student_error"


def ask_student(client, model, text):
    """Ask the small model about one ticket. It never crashes: a problem is reported in the result."""
    # Step 1: call the model. If the call itself fails (for example the server is off),
    # report that as a problem instead of crashing.
    try:
        reply = client.chat.completions.create(
            model=model,
            messages=build_messages(text),
            temperature=0,
            logprobs=True,  # ask the server to also tell us how likely each token was
            max_tokens=700,
        )
    except Exception as error:
        log.warning("student call failed: %s", str(error)[:200])
        return StudentAnswer(None, None, "student_error")

    choice = reply.choices[0]
    answer_text = choice.message.content
    if answer_text is None:
        answer_text = ""

    # Step 2: check that the answer is valid JSON that fits our schema.
    try:
        result = TicketTriage.model_validate_json(answer_text)
    except ValidationError:
        return StudentAnswer(None, None, "invalid_output")

    # Step 3: collect the token probabilities, if the server sent any.
    tokens = []
    if choice.logprobs is not None:
        if choice.logprobs.content is not None:
            for item in choice.logprobs.content:
                tokens.append((item.token, item.logprob))

    # Step 4: turn the probabilities into one confidence number.
    confidences = field_confidences(answer_text, tokens)
    confidence = overall_confidence(confidences)
    return StudentAnswer(result, confidence, None)


class TriageService:
    def __init__(
        self,
        student_client,
        student_model,
        metrics,
        min_confidence=0.0,
        teacher_client=None,
        teacher_endpoint=None,
        teacher_model=None,
        teacher_budget=0,
    ):
        self.student_client = student_client
        self.student_model = student_model
        self.metrics = metrics
        self.min_confidence = min_confidence  # answers the student is less sure of than this go to the teacher
        self.teacher_client = teacher_client
        self.teacher_endpoint = teacher_endpoint
        self.teacher_model = teacher_model
        self.teacher_budget = teacher_budget  # paid calls left. 0 means: never call the teacher

    def can_use_teacher(self):
        """True only if a teacher is set up AND there are paid calls left."""
        if self.teacher_client is None:
            return False
        if self.teacher_budget <= 0:
            return False
        return True

    def triage(self, text, request_id=None):
        """Triage one ticket. Returns a TriageResponse, or raises GatewayError."""
        # [:12] means "the first 12 characters", which is enough to tell requests apart.
        if request_id is None:
            request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()

        # Step 1: ask the small model.
        answer = ask_student(self.student_client, self.student_model, text)

        # Step 2: decide whether its answer can be trusted. `reason` stays None if it can.
        reason = answer.problem
        if reason is None:
            if answer.confidence is not None:
                if answer.confidence < self.min_confidence:
                    reason = "low_confidence"

        # Step 3: trusted, so we are done. The teacher costs nothing here.
        if reason is None:
            return self.finish(request_id, "student", answer.result, answer.confidence, None, started)

        # Step 4: not trusted. Try the teacher, if we are allowed to.
        if self.can_use_teacher():
            teacher_result = self.ask_teacher(text)
            if teacher_result is not None:
                return self.finish(request_id, "teacher", teacher_result, answer.confidence, reason, started)

        # Step 5: no teacher available. A valid but unsure answer is better than no answer.
        if answer.result is not None:
            log.info("id=%s unsure, but no teacher is available: answering with the student", request_id)
            return self.finish(request_id, "student", answer.result, answer.confidence, None, started)

        # Step 6: nothing usable at all.
        self.metrics.record_failure()
        raise GatewayError("No usable answer (" + reason + "), and the teacher is not available.")

    def ask_teacher(self, text):
        """Ask the paid teacher. Returns the answer, or None if the call failed."""
        # Count the call BEFORE making it: a call that fails halfway may still be charged.
        self.teacher_budget = self.teacher_budget - 1
        try:
            result, details = label_one(self.teacher_client, self.teacher_endpoint, self.teacher_model, text)
            return result
        except Exception as error:
            log.warning("teacher call failed: %s", str(error)[:200])
            return None

    def finish(self, request_id, source, result, confidence, fallback_reason, started):
        """Record the answer in the metrics, log it, and build the response."""
        latency_ms = int((time.perf_counter() - started) * 1000)
        self.metrics.record_answer(source, latency_ms, fallback_reason)

        # Round the confidence for the log line (None stays None).
        confidence_for_log = confidence
        if confidence is not None:
            confidence_for_log = round(confidence, 3)
        log.info(
            "id=%s source=%s reason=%s confidence=%s latency_ms=%d",
            request_id,
            source,
            fallback_reason,
            confidence_for_log,
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
    student_config="student_q4",
    use_teacher=False,
    approved=False,
    teacher_budget=0,
    min_confidence=0.0,
    teacher_cost_per_ticket=0.002,
):
    """Connect the real models from configs/models.yaml and return a ready TriageService."""
    # By default there is no teacher at all.
    teacher_client = None
    teacher_endpoint = None
    teacher_model = None
    budget = 0

    # Safety checks for the paid teacher come FIRST, before anything is connected.
    if use_teacher:
        if teacher_budget <= 0:
            raise UserError("--teacher needs a positive --teacher-budget (the most paid calls allowed).")
        teacher_endpoint = load_endpoint("teacher")
        require_approval_for_paid_calls(teacher_endpoint, teacher_budget, approved)
        teacher_client = make_client(teacher_endpoint, max_retries=2, timeout=60)
        teacher_model = teacher_endpoint.model
        budget = teacher_budget

    student_endpoint = load_endpoint(student_config)
    student_client = make_client(student_endpoint, max_retries=0, timeout=60)
    student_model = student_endpoint.model
    if student_model is None:
        student_model = ""

    metrics = GatewayMetrics(teacher_cost_per_ticket)
    return TriageService(
        student_client=student_client,
        student_model=student_model,
        metrics=metrics,
        min_confidence=min_confidence,
        teacher_client=teacher_client,
        teacher_endpoint=teacher_endpoint,
        teacher_model=teacher_model,
        teacher_budget=budget,
    )
