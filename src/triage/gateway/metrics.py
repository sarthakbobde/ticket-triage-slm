"""Counters the gateway keeps about itself, shown at GET /metrics.

Everything lives in memory and starts again from zero when the server restarts. That is enough to
answer the questions Step 12 asks: how often did we need the teacher, how fast were we, and how much
money did the small model save?
"""

import threading

from triage.evals.metrics import percentile


class GatewayMetrics:
    def __init__(self, teacher_cost_per_ticket, window=1000):
        # What one teacher call costs. We measured this in Step 08: about 625 tokens in and 95 out per
        # ticket, which is about $0.002 at the assumed gpt-4.1 prices (not yet checked against the bill).
        self.teacher_cost_per_ticket = teacher_cost_per_ticket

        # Only the most recent `window` response times are kept, so memory does not grow forever.
        self.window = window

        # Several requests can arrive at the same moment. The lock makes them take turns when they
        # update the numbers, so two requests cannot overwrite each other's changes.
        self.lock = threading.Lock()

        self.student_count = 0
        self.teacher_count = 0
        self.failure_count = 0
        self.fallback_reasons = {}  # reason -> how many times it happened
        self.student_latencies = []
        self.teacher_latencies = []

    def record_answer(self, source, latency_ms, fallback_reason):
        """Remember one successful answer."""
        with self.lock:
            if source == "student":
                self.student_count = self.student_count + 1
                self.student_latencies.append(latency_ms)
                self.student_latencies = self.student_latencies[-self.window :]  # keep the last `window`
            else:
                self.teacher_count = self.teacher_count + 1
                self.teacher_latencies.append(latency_ms)
                self.teacher_latencies = self.teacher_latencies[-self.window :]

            if fallback_reason is not None:
                times_before = self.fallback_reasons.get(fallback_reason, 0)
                self.fallback_reasons[fallback_reason] = times_before + 1

    def record_failure(self):
        """Remember one request that got no usable answer."""
        with self.lock:
            self.failure_count = self.failure_count + 1

    def snapshot(self):
        """A copy of all the numbers, as a dictionary ready to be sent as JSON."""
        with self.lock:
            answered = self.student_count + self.teacher_count

            # Share of answers that needed the teacher. None if there were no answers yet.
            fallback_rate = None
            if answered > 0:
                fallback_rate = self.teacher_count / answered

            all_latencies = self.student_latencies + self.teacher_latencies

            latency = {}
            latency["p50"] = percentile(all_latencies, 50)  # half of the requests were faster than this
            latency["p95"] = percentile(all_latencies, 95)  # 95 out of 100 were faster than this
            latency["student_p50"] = percentile(self.student_latencies, 50)
            latency["teacher_p50"] = percentile(self.teacher_latencies, 50)

            data = {}
            data["requests_answered"] = answered
            data["requests_failed"] = self.failure_count
            data["answered_by_student"] = self.student_count
            data["answered_by_teacher"] = self.teacher_count
            data["fallback_rate"] = fallback_rate
            data["fallback_reasons"] = dict(self.fallback_reasons)
            data["latency_ms"] = latency
            # Every ticket the small model answered is a ticket we did not pay the teacher for.
            data["estimated_cost_saved_usd"] = round(self.student_count * self.teacher_cost_per_ticket, 4)
            return data
