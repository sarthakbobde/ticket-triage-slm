"""Prompts shared by the teacher, the small model, and the evaluation code.

The SAME system prompt is used for labelling and for training/serving. If the small model is
trained on one prompt and served with another, quality silently drops.

CATEGORY_DEFINITIONS are the labelling guidelines: they define what "correct" means (ADR-010).
Changing them changes the truth, so it needs a new ADR and a SCHEMA_VERSION bump.
"""

from triage.schema import Category, Priority, Sentiment

CATEGORY_DEFINITIONS = {
    Category.TECHNICAL: "something is broken or not working: errors, crashes, bugs, slowness, "
    "integration failures, security incidents affecting one customer",
    Category.PRODUCT: "nothing is broken: how to use or configure a product, what it can do, "
    "compatibility, best practices, feature requests",
    Category.BILLING: "invoices, charges, payments, subscriptions, refunds of a charge",
    Category.OUTAGE: "a service is down or degraded for many users, or planned maintenance",
    Category.CUSTOMER_SERVICE: "account changes, orders, returns/exchanges, pre-sales and pricing "
    "questions, general enquiries, internal employee matters (payroll, onboarding, leave, "
    "staff policies), and complaints that are not a technical fault",
}

# The lists of allowed words, and one line of explanation per category, to put into the prompt.
category_names = []
category_lines = []
for category in Category:
    category_names.append(category.value)
    definition = CATEGORY_DEFINITIONS[category]
    category_lines.append(f"    {category.value:<16} = {definition}")
_category_lines = "\n".join(category_lines)

priority_names = []
for priority in Priority:
    priority_names.append(priority.value)

sentiment_names = []
for sentiment in Sentiment:
    sentiment_names.append(sentiment.value)

SYSTEM_PROMPT = f"""You are a customer-support triage assistant.
Read the ticket and respond with ONLY a JSON object, no prose, with these fields:

- category: one of {category_names}
{_category_lines}
- priority: one of {priority_names}
    high   = customer blocked, no workaround
    medium = problem with a workaround, or time-sensitive question
    low    = general question, feedback, feature request
- sentiment: one of {sentiment_names}
- products: list of products, software, services or devices explicitly named in the ticket
  (e.g. ["Jira", "MacBook Pro"]); [] if none; never invent names
- summary: one sentence for the support agent
- suggested_reply: a polite reply to the customer, 2-4 sentences, under 500 characters;
  do not promise refunds or timelines
"""


def build_messages(ticket_text):
    """The two messages sent to a model for one ticket: the instructions, then the ticket."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ticket:\n{ticket_text.strip()}"},
    ]
