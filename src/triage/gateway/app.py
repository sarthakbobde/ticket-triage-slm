"""The HTTP API (Step 11).

    POST /triage    {"text": "..."}  ->  the triage answer, who wrote it, how sure, how long it took
    GET  /health                     ->  is the server up?
    GET  /metrics                    ->  how often the teacher was needed, speed, money saved

Start it:  uv run triage serve
Try it:    curl -X POST localhost:8000/triage -H "Content-Type: application/json" -d '{"text": "..."}'

How to read the lines that start with @app: a line like  @app.post("/triage")  is a label put on the
function right below it. It means: "when a POST request arrives at the address /triage, run this
function". This is how the web framework (FastAPI) is told which function handles which address.
"""

from fastapi import FastAPI, HTTPException, Request

from triage.gateway.schemas import TriageRequest, TriageResponse
from triage.gateway.service import GatewayError


def create_app(service):
    """Build the web application around a TriageService."""
    app = FastAPI(title="Ticket triage gateway", version="0.1.0")

    @app.post("/triage", response_model=TriageResponse)
    def triage(body: TriageRequest, request: Request):
        # An empty ticket is not a ticket. 422 is the standard code for "your data is wrong".
        if body.text.strip() == "":
            raise HTTPException(status_code=422, detail="text must not be empty")

        # The caller may send its own request id in a header, so it can find the request in the logs.
        request_id = request.headers.get("x-request-id")

        try:
            return service.triage(body.text, request_id=request_id)
        except GatewayError as error:
            # 503 means "service unavailable": we could not produce an answer right now.
            # "from error" keeps the original error attached, so the logs show what really went wrong.
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/health")
    def health():
        answer = {}
        answer["status"] = "ok"
        answer["student_model"] = service.student_model
        answer["teacher_available"] = service.can_use_teacher()
        return answer

    @app.get("/metrics")
    def metrics():
        numbers = service.metrics.snapshot()
        numbers["teacher_budget_left"] = service.teacher_budget
        return numbers

    return app
