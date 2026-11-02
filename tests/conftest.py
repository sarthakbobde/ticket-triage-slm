"""Shared test helpers. No test here touches the network, a GPU, or a paid API."""

import json
import math
import re
from types import SimpleNamespace

import pytest


def valid_answer(category="technical", priority="high", sentiment="negative"):
    """A model answer (as raw JSON text) that fits our schema. Change the three decisions as needed."""
    data = {}
    data["category"] = category
    data["priority"] = priority
    data["sentiment"] = sentiment
    data["products"] = ["Jira"]
    data["summary"] = "Customer cannot log in."
    data["suggested_reply"] = "Thanks, we are looking into it."
    return json.dumps(data)


def label_dict(category="technical", priority="high", sentiment="negative"):
    """A label as stored in the teacher's output files (a dictionary)."""
    return json.loads(valid_answer(category, priority, sentiment))


class FakeClient:
    """Stands in for the real model client, so tests never call a real model.

    `answers` is a list of what the fake model replies, one item per call, in order.
    When the list runs out, the last item is used again. An item can be:
      - a text: the model's reply
      - an Exception: raised instead of a reply, to pretend the call failed
      - a pair (text, tokens): a reply together with token probabilities (see tokens_for below)
    """

    def __init__(self, answers):
        self.answers = answers
        self.calls = 0  # how many times the fake model was asked
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(
        self,
        model=None,
        messages=None,
        temperature=None,
        response_format=None,
        logprobs=None,
        max_tokens=None,
        reasoning_effort=None,
    ):
        # Pick the next answer (or the last one, if we ran out).
        position = self.calls
        if position > len(self.answers) - 1:
            position = len(self.answers) - 1
        answer = self.answers[position]
        self.calls = self.calls + 1

        if isinstance(answer, Exception):
            raise answer

        tokens = None
        if isinstance(answer, tuple):
            answer, tokens = answer  # a pair: the text and its token probabilities

        message = SimpleNamespace(content=answer)

        token_info = None
        if tokens is not None:
            items = []
            for text, logprob in tokens:
                items.append(SimpleNamespace(token=text, logprob=logprob))
            token_info = SimpleNamespace(content=items)

        choice = SimpleNamespace(message=message, logprobs=token_info)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def fake_client_factory():
    return FakeClient


def tokens_for(content, confidence=None):
    """Cut an answer into (token text, log-probability) pairs, the way a model reports them.

    Each of the three decision values (category, priority, sentiment) becomes its own token. Its
    probability comes from `confidence` (a dictionary such as {"priority": 0.5}); the default is 1.0.
    Every other piece of text is certain (probability 1).
    """
    if confidence is None:
        confidence = {}

    # Find where each decision value starts and ends inside the text.
    pieces = []
    for field in ["category", "priority", "sentiment"]:
        match = re.search('"' + field + '"\\s*:\\s*"([^"]*)"', content)
        start = match.start(1)
        end = match.end(1)
        probability = confidence.get(field, 1.0)
        pieces.append((start, end, probability))
    pieces.sort()

    # Walk through the text, making a certain token for the text between values and a token for each value.
    tokens = []
    cursor = 0
    for start, end, probability in pieces:
        if start > cursor:
            tokens.append((content[cursor:start], 0.0))  # log of 1.0 is 0.0
        tokens.append((content[start:end], math.log(probability)))
        cursor = end
    if cursor < len(content):
        tokens.append((content[cursor:], 0.0))
    return tokens
