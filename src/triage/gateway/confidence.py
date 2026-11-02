"""How sure was the student of its answer? (Step 11)

A language model writes its answer one token (word piece) at a time, and for each token it can tell
us the probability it gave to the token it chose. For the JSON answer

    {"category": "technical", "priority": "high", ...}

the token `technical` might have probability 1.00 (certain) and `high` 0.60 (it nearly said
`medium`). The probability of those chosen values is a cheap confidence signal: when the model
hesitates between two categories, the gateway can hand the ticket to the stronger teacher model.

Only the three decision fields are used. The summary and reply are free text, and a long sentence
always contains some unlikely words, which says nothing about whether the triage is right.
"""

import math
import re
from collections.abc import Sequence

FIELDS = ("category", "priority", "sentiment")


def field_confidences(content: str, tokens: Sequence[tuple[str, float]]) -> dict[str, float]:
    """The probability of the chosen value for each decision field.

    `tokens` is the answer as (token text, log-probability) pairs, in order. Their texts joined
    must equal `content`, otherwise we cannot line them up and return nothing.
    """
    if "".join(text for text, _ in tokens) != content:
        return {}

    # Where each token starts and ends inside the answer text.
    spans = []
    position = 0
    for text, logprob in tokens:
        spans.append((position, position + len(text), math.exp(logprob)))
        position = position + len(text)

    confidences = {}
    for field in FIELDS:
        match = re.search(rf'"{field}"\s*:\s*"', content)
        if match is None:
            continue
        value_start = match.end()  # the first character of the value, e.g. the "t" of "technical"
        for start, end, probability in spans:
            if start <= value_start < end:
                confidences[field] = probability
                break
    return confidences


def overall_confidence(confidences: dict[str, float]) -> float | None:
    """One number for the whole answer: its weakest decision. None if we know nothing."""
    if not confidences:
        return None
    return min(confidences.values())
