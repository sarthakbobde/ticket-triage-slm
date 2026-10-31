"""The output contract. Every component validates against this.

See docs/DECISIONS.md#adr-001 for the reasoning, ADR-010 / ADR-011 for v1.0 changes and
ADR-013 for v1.1 (hr merged into customer_service).

Changing anything here means bumping SCHEMA_VERSION and re-labelling data.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.1"


class Category(str, Enum):
    # Definitions live in prompts.py (CATEGORY_DEFINITIONS); they are the labelling guidelines.
    TECHNICAL = "technical"
    PRODUCT = "product"
    BILLING = "billing"
    OUTAGE = "outage"
    CUSTOMER_SERVICE = "customer_service"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class TicketTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    sentiment: Sentiment
    products: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Products, software, services or devices explicitly named in the ticket.",
    )
    summary: str = Field(max_length=200, description="One sentence, agent-facing.")
    suggested_reply: str = Field(max_length=800, description="Customer-facing draft reply.")


if __name__ == "__main__":
    # Quick look at what the model will be asked to produce.
    import json

    print(json.dumps(TicketTriage.model_json_schema(), indent=2))
