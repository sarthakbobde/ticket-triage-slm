"""How sure was the small model of its answer? (Step 11)

A language model writes its answer one small piece at a time. A piece is called a "token" (often a
word or part of a word). For every token the server can tell us the probability the model gave to
the token it chose.

Take this answer:

    {"category": "technical", "priority": "high", ...}

Say the token `technical` had probability 1.00 (completely certain) and the token `high` had
probability 0.60 (the model nearly wrote `medium` instead). Those probabilities tell us how sure the
model was. When it hesitates, the gateway can hand the ticket to the stronger teacher model.

We only look at the three decision fields: category, priority and sentiment. The summary and the
reply are free text, and a long sentence always contains a few unlikely words. That says nothing
about whether the triage decision is right.
"""

import math
import re

DECISION_FIELDS = ["category", "priority", "sentiment"]


def find_token_positions(tokens):
    """For every token, work out where it starts and ends inside the full answer text.

    `tokens` is a list of pairs: (the token's text, the log-probability the model gave it).
    Returns a list of triples: (start, end, probability).
    """
    positions = []
    start = 0
    for token_text, log_probability in tokens:
        end = start + len(token_text)
        probability = math.exp(log_probability)  # log-probability -> normal probability (0 to 1)
        positions.append((start, end, probability))
        start = end
    return positions


def field_confidences(answer_text, tokens):
    """The probability of the chosen value for each decision field.

    Returns a dictionary such as {"category": 1.0, "priority": 0.6, "sentiment": 0.99}.
    Returns an empty dictionary if the tokens do not add up to the answer text.
    """
    # Glue all the tokens back together. The result must equal the answer, otherwise we cannot
    # line the tokens up with the text and we should not guess.
    joined = ""
    for token_text, log_probability in tokens:
        joined = joined + token_text
    if joined != answer_text:
        return {}

    positions = find_token_positions(tokens)

    confidences = {}
    for field in DECISION_FIELDS:
        # Find the place in the text where this field's value begins. For the field "category"
        # we look for:  "category": "   (a quote, the name, a quote, a colon, optional spaces, a quote)
        pattern = '"' + field + '"\\s*:\\s*"'
        match = re.search(pattern, answer_text)
        if match is None:
            continue
        value_start = match.end()  # the position of the first letter of the value, e.g. the "t" of "technical"

        # Find the token that contains that position and use its probability.
        for start, end, probability in positions:
            if start <= value_start and value_start < end:
                confidences[field] = probability
                break
    return confidences


def overall_confidence(confidences):
    """One number for the whole answer: the weakest of its decisions (None if we know nothing)."""
    # No probabilities were available, so we do not know how sure the model was.
    if len(confidences) == 0:
        return None

    # Start with the highest possible value, then keep the smallest one we see.
    weakest = 1.0
    for field in confidences:
        if confidences[field] < weakest:
            weakest = confidences[field]
    return weakest
