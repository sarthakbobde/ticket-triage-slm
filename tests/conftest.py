"""Shared test helpers. No test here touches the network, a GPU, or a paid API."""

import json
from types import SimpleNamespace

import pytest


def valid_answer(**overrides) -> str:
    """A model answer (raw JSON text) that fits the schema. Override fields to break it."""
    data = {
        "category": "technical",
        "priority": "high",
        "sentiment": "negative",
        "products": ["Jira"],
        "summary": "Customer cannot log in.",
        "suggested_reply": "Thanks, we are looking into it.",
    }
    data.update(overrides)
    return json.dumps(data)


def label_dict(**overrides) -> dict:
    """A label as stored in the teacher's output files."""
    return json.loads(valid_answer(**overrides))


class FakeClient:
    """Stands in for the OpenAI client. `answers` is replayed in order (the last one repeats).

    An Exception in the list is raised instead of returned, to simulate a failed call.
    """

    def __init__(self, answers: list):
        self.answers = answers
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        tokens = None
        if isinstance(answer, tuple):  # (text, [(token, logprob), ...])
            answer, tokens = answer
        message = SimpleNamespace(content=answer)
        logprobs = None
        if tokens is not None:
            items = [SimpleNamespace(token=text, logprob=logprob) for text, logprob in tokens]
            logprobs = SimpleNamespace(content=items)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, logprobs=logprobs)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


@pytest.fixture
def fake_client_factory():
    return FakeClient


def tokens_for(content: str, confidence: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Cut an answer into (token, log-probability) pairs the way a model would report them.

    Each decision value (category, priority, sentiment) becomes its own token. Its probability comes
    from `confidence` (default 1.0); every other token is certain (probability 1).
    """
    import math
    import re

    confidence = confidence or {}
    pieces = []
    cursor = 0
    for field in ("category", "priority", "sentiment"):
        match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', content)
        start, end = match.start(1), match.end(1)
        pieces.append((start, end, confidence.get(field, 1.0)))
    pieces.sort()
    tokens = []
    for start, end, probability in pieces:
        if start > cursor:
            tokens.append((content[cursor:start], 0.0))
        tokens.append((content[start:end], math.log(probability)))
        cursor = end
    if cursor < len(content):
        tokens.append((content[cursor:], 0.0))
    return tokens
