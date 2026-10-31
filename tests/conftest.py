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
        message = SimpleNamespace(content=answer)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


@pytest.fixture
def fake_client_factory():
    return FakeClient
