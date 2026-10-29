"""The output contract. Every component validates against this.

See docs/DECISIONS.md#adr-001 for the reasoning.

Changing anything here means bumping SCHEMA_VERSION and re-labelling data.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1-draft"  # becomes 1.0 once categories are finalised in Step 02


class Category(str, Enum):
    # DRAFT: finalise after exploring the dataset in Step 02.
    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    TECHNICAL_ISSUE = "technical_issue"
    SHIPPING_DELIVERY = "shipping_delivery"
    REFUND_RETURN = "refund_return"
    PRODUCT_INQUIRY = "product_inquiry"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Sentiment(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class Entities(BaseModel):
    """Facts pulled from the ticket. null when not mentioned — never guessed."""

    model_config = ConfigDict(extra="forbid")

    order_id: str | None = None
    product: str | None = None
    account_email: str | None = None
    amount: str | None = None


class TicketTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    sentiment: Sentiment
    entities: Entities
    summary: str = Field(max_length=200, description="One sentence, agent-facing.")
    suggested_reply: str = Field(max_length=1200, description="Customer-facing draft reply.")


if __name__ == "__main__":
    # Quick look at what the model will be asked to produce.
    import json

    print(json.dumps(TicketTriage.model_json_schema(), indent=2))
