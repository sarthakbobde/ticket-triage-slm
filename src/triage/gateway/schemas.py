"""The shapes of the data going into and out of the HTTP API.

A "schema" like this is a form with named boxes. The web framework checks that incoming data fits
the form, and turns our Python objects into JSON on the way out.
"""

from typing import Optional

from pydantic import BaseModel

from triage.schema import TicketTriage


class TriageRequest(BaseModel):
    # The support ticket: subject and body as one piece of text.
    text: str


class TriageResponse(BaseModel):
    request_id: str  # a short id so one request can be found again in the logs
    source: str  # "student" or "teacher": which model wrote the answer
    result: TicketTriage  # the triage answer itself
    # Optional[float] means "a number, or None". None = the server did not tell us how sure it was.
    confidence: Optional[float]
    # Why the teacher was used instead of the student. None = the student answered.
    fallback_reason: Optional[str]
    latency_ms: int  # how long this request took, in milliseconds
