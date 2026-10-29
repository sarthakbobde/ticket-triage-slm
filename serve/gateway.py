"""Step 11 — FastAPI gateway in front of the local student model.

TODO (Step 11):
- POST /triage {text} -> TicketTriage
- call student; validate with Pydantic; on failure or low confidence -> fall back to teacher
- log request id, latency, which model answered, token counts
- GET /metrics: fallback rate, latency percentiles, estimated cost saved
"""
