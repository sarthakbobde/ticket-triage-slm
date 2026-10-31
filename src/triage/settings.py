"""Model endpoints from configs/models.yaml, checked when loaded.

Every model (teacher, base models, the served student) is an OpenAI-compatible endpoint
(ADR-004), so switching models is a config edit, not a code change.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from triage import paths
from triage.errors import UserError


class Price(BaseModel):
    """US dollars per 1 million tokens. Used only for cost estimates."""

    input: float
    output: float


class ModelEndpoint(BaseModel):
    # A typo in the YAML (e.g. "base_ulr") is an error instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str | None = None
    api_key_env: str | None = None  # NAME of the environment variable holding the key
    temperature: float = 0
    reasoning_effort: str | None = None
    response_format: str = "json_schema"
    rpm: int | None = None  # max requests per minute, None = no limit
    candidates: dict[str, Price] = {}


def load_endpoint(name: str, config_path: Path = paths.MODELS_CONFIG) -> ModelEndpoint:
    """Load one section of models.yaml, e.g. "teacher" or "base_qwen"."""
    sections = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if name not in sections:
        raise UserError(f"No '{name}' section in {config_path.name}. Available: {list(sections)}")
    try:
        return ModelEndpoint(**sections[name])
    except ValidationError as error:
        raise UserError(f"Bad '{name}' section in {config_path.name}:\n{error}") from error
