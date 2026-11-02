"""The HTTP API (Step 11).

    POST /triage    {"text": "..."}  ->  the triage JSON, who answered, how sure, how long it took
    GET  /health                     ->  is the server up
    GET  /metrics                    ->  fallback rate, latency, money saved

Run it with:  uv run triage serve
Try it:       curl -X POST localhost:8000/triage -H "Content-Type: application/json" -d '{"text": "..."}'
"""

from fastapi import FastAPI, Header, HTTPException

from triage.gateway.schemas import TriageRequest, TriageResponse
from triage.gateway.service import GatewayError, TriageService


def create_app(service: TriageService) -> FastAPI:
    app = FastAPI(title="Ticket triage gateway", version="0.1.0")

    # A plain `def` endpoint: FastAPI runs it in a worker thread, so a slow model call does not
    # block the other requests.
    @app.post("/triage", response_model=TriageResponse)
    def triage(request: TriageRequest, x_request_id: str | None = Header(default=None)) -> TriageResponse:
        try:
            return service.triage(request.text, request_id=x_request_id)
        except GatewayError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "student_model": service.student_model,
            "teacher_available": service.can_use_teacher(),
        }

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return {**service.metrics.snapshot(), "teacher_budget_left": service.teacher_budget}

    return app
