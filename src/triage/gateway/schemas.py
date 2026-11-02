"""The request and response shapes of the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field

from triage.schema import TicketTriage


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000, description="The support ticket, subject and body.")


class TriageResponse(BaseModel):
    request_id: str
    source: Literal["student", "teacher"] = Field(description="Which model produced the answer.")
    result: TicketTriage
    confidence: float | None = Field(
        description="The student's own certainty, 0 to 1 (None if the server gave no probabilities).",
    )
    fallback_reason: str | None = Field(
        description="Why the teacher was used instead of the student, or None if the student answered.",
    )
    latency_ms: int
