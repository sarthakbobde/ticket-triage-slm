"""Prompts shared by the teacher, the student, and the eval harness.

The SAME system prompt is used for labelling and for training/inference. If the student is
trained on one prompt and served with another, quality silently drops.
"""

from triage.schema import Category, Priority, Sentiment

SYSTEM_PROMPT = f"""You are a customer-support triage assistant.
Read the ticket and respond with ONLY a JSON object, no prose, with these fields:

- category: one of {[c.value for c in Category]}
- priority: one of {[p.value for p in Priority]}
    urgent = service down / security / money lost now
    high   = customer blocked, no workaround
    medium = problem with a workaround, or time-sensitive question
    low    = general question, feedback, feature request
- sentiment: one of {[s.value for s in Sentiment]}
- entities: {{"order_id", "product", "account_email", "amount"}} — use null if not stated; never invent values
- summary: one sentence for the support agent
- suggested_reply: a short, polite reply to the customer; do not promise refunds or timelines
"""


def build_messages(ticket_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ticket:\n{ticket_text.strip()}"},
    ]
