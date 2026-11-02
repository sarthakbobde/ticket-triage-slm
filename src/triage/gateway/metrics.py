"""Counters the gateway keeps about itself, shown at GET /metrics.

Everything lives in memory and resets when the server restarts. That is enough to answer the
questions Step 12 asks: how often did we fall back to the teacher, how fast were we, and how much
money did the small model save?
"""

import threading
from collections import Counter, deque
from typing import Any

from triage.evals.metrics import percentile


class GatewayMetrics:
    def __init__(self, teacher_cost_per_ticket: float, window: int = 1000):
        # What one teacher call costs. Measured in Step 08: about 625 tokens in and 95 out per
        # ticket, which is $0.002 at the assumed gpt-4.1 prices (unverified).
        self.teacher_cost_per_ticket = teacher_cost_per_ticket
        self._lock = threading.Lock()
        self._answered = Counter()  # by source: "student" / "teacher"
        self._fallback_reasons = Counter()
        self._failures = 0
        self._latencies = {"student": deque(maxlen=window), "teacher": deque(maxlen=window)}

    def record_answer(self, source: str, latency_ms: int, fallback_reason: str | None) -> None:
        with self._lock:
            self._answered[source] += 1
            self._latencies[source].append(latency_ms)
            if fallback_reason:
                self._fallback_reasons[fallback_reason] += 1

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            student = self._answered["student"]
            teacher = self._answered["teacher"]
            answered = student + teacher
            everything = list(self._latencies["student"]) + list(self._latencies["teacher"])
            return {
                "requests_answered": answered,
                "requests_failed": self._failures,
                "answered_by_student": student,
                "answered_by_teacher": teacher,
                "fallback_rate": (teacher / answered) if answered else None,
                "fallback_reasons": dict(self._fallback_reasons),
                "latency_ms": {
                    "p50": percentile(everything, 50),
                    "p95": percentile(everything, 95),
                    "student_p50": percentile(list(self._latencies["student"]), 50),
                    "teacher_p50": percentile(list(self._latencies["teacher"]), 50),
                },
                # Tickets the small model answered are tickets we did not pay the teacher for.
                "estimated_cost_saved_usd": round(student * self.teacher_cost_per_ticket, 4),
            }
