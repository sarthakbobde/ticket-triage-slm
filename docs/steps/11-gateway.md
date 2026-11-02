# Step 11: The gateway

**Who:** 🤝 · **Status:** done · **Cost:** $0 to build and calibrate (the teacher fallback is off by default)

## The idea
A small web service in front of the model. Callers send a ticket and get triage JSON back. The
gateway asks the small model first, checks the answer, and can hand a ticket to the paid teacher
when the answer is unusable or the small model was unsure.

```
POST /triage {"text": "..."}
      |
   small model (local) ---> valid JSON?  sure enough? ---yes---> answer (source: student)
                                      | no
                                      +--> teacher (paid; only if enabled and budget is left) ---> answer (source: teacher)
```

## Run it
```bash
uv sync --extra serve
uv run triage serve                                           # student only: free, on this machine only
uv run triage serve --teacher --teacher-budget 50 --yes       # allow at most 50 paid teacher calls
uv run triage serve --min-confidence 0.7                      # also route unsure answers to the teacher
```
Try it:
```bash
curl -X POST localhost:8000/triage -H "Content-Type: application/json" -d "{\"text\": \"The login page shows error 500\"}"
curl localhost:8000/metrics
```
The reply holds the triage answer plus `source`, `confidence`, `fallback_reason`, `latency_ms` and a `request_id`.

## How confidence works
The server reports the probability of every token it writes. The answer
`{"category": "technical", "priority": "high", ...}` is made of tokens, and the token `high` might
carry probability 0.60. The gateway takes the probabilities of the three decision values (category,
priority, sentiment) and uses the **weakest** as the confidence of the whole answer. Asking for the
probabilities does not slow the model down (measured).

## Safety rails
- The teacher is **off by default**. Turning it on needs `--teacher`, `--teacher-budget N` and `--yes`.
- The budget is a hard cap on paid calls; a call that fails still counts.
- The server listens on `127.0.0.1` only.
- If nothing usable can be produced the caller gets a clean `503`, never a crash.
- Tests use fake models, so they can never cost money.

## Calibration: what threshold should we use?
`uv run triage eval calibrate` asks the small model about the 200 gold tickets, records its
confidence, then simulates the gateway at each threshold using the teacher answers we already have
(free). Below the threshold a ticket takes the teacher's answer.

| Send to teacher below | Tickets sent | Category | Priority | Sentiment |
|---|---|---|---|---|
| 0.00 (never) | 0% | 77.5% | 67.0% | 80.0% |
| 0.60 | 10% | 78.5% | 68.5% | 78.5% |
| 0.70 | 24% | 78.5% | 67.0% | 76.5% |
| 0.80 | 40% | 79.5% | 66.0% | 74.0% |
| 0.90 | 56% | 79.5% | 68.0% | 72.0% |
| 0.99 | 84% | 80.0% | 68.5% | 70.5% |

Findings:
1. **Confidence is a real signal about the teacher.** When the student agrees with the teacher its
   average confidence is 0.86; when they differ it is 0.71. Below 0.7, the gateway would send 24% of
   tickets and catch 52% of the teacher disagreements.
2. **But sending tickets to the teacher barely improves accuracy against the gold labels:** at most
   +2.5 points on category (5 tickets) and none on priority. **Sentiment gets worse** (80% -> 70.5%),
   because the teacher is worse than the student on sentiment here.
3. Confidence is almost the same for right and wrong answers against gold (0.84 vs 0.79): most
   "errors" are the student confidently copying the teacher where the gold labels disagree
   (Step 08). A confidence score cannot detect a disagreement about the definition of a category.
4. 200 tickets cannot separate differences of 1-3 points (2-6 tickets).

Decision and reasons: [ADR-017](../DECISIONS.md#adr-017).

## What is not done
- **Streaming and concurrency limits:** requests are answered one by one by the single GPU.
- **Persistent metrics:** `/metrics` resets when the server restarts.
- **Authentication:** the server is for this machine only.
