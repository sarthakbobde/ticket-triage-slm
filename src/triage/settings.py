"""Model settings from configs/models.yaml, checked when they are loaded.

Every model (the teacher, the base models, the small model we serve) is reached through an
OpenAI-style web address. So switching models means editing that file, not the code.
"""

from typing import Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from triage import paths
from triage.errors import UserError


class Price(BaseModel):
    """US dollars per 1 million tokens. Only used for cost estimates."""

    input: float
    output: float


class ModelEndpoint(BaseModel):
    # A typo in the YAML file (for example "base_ulr") becomes an error instead of being ignored.
    model_config = ConfigDict(extra="forbid")

    # (Optional[...] means "this value, or None".)
    base_url: str
    model: Optional[str] = None
    api_key_env: Optional[str] = None  # the NAME of the environment variable that holds the key
    temperature: float = 0
    reasoning_effort: Optional[str] = None
    response_format: str = "json_schema"
    rpm: Optional[int] = None  # most requests per minute. None = no limit
    candidates: Dict[str, Price] = {}


def load_endpoint(name, config_path=paths.MODELS_CONFIG):
    """Load one section of models.yaml, for example "teacher" or "base_qwen"."""
    sections = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if name not in sections:
        available = list(sections.keys())
        raise UserError("No '" + name + "' section in " + config_path.name + ". Available: " + str(available))

    try:
        # model_validate checks the section and builds the ModelEndpoint from it.
        return ModelEndpoint.model_validate(sections[name])
    except ValidationError as error:
        raise UserError("Bad '" + name + "' section in " + config_path.name + ":\n" + str(error)) from error
