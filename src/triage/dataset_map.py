"""Map the dataset's own labels onto our schema (ADR-010).

These mapped labels are a CROSS-CHECK only: we measure how often the teacher agrees with them.
They are never training targets, because the EDA showed the dataset's queue labels are noisy.
"""

from triage.schema import Category

QUEUE_TO_CATEGORY = {
    "Technical Support": Category.TECHNICAL,
    "IT Support": Category.TECHNICAL,
    "Product Support": Category.PRODUCT,
    "Billing and Payments": Category.BILLING,
    "Service Outages and Maintenance": Category.OUTAGE,
    "Customer Service": Category.CUSTOMER_SERVICE,
    "Sales and Pre-Sales": Category.CUSTOMER_SERVICE,
    "General Inquiry": Category.CUSTOMER_SERVICE,
    "Returns and Exchanges": Category.CUSTOMER_SERVICE,
    "Human Resources": Category.CUSTOMER_SERVICE,  # hr merged in, ADR-013
}
