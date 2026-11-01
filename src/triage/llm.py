"""Talking to model endpoints: building the client and the JSON-format request option."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from triage import paths
from triage.errors import UserError
from triage.schema import TicketTriage
from triage.settings import ModelEndpoint

log = logging.getLogger(__name__)


def is_local_url(base_url: str) -> bool:
    return "localhost" in base_url or "127.0.0.1" in base_url


def require_approval_for_paid_calls(endpoint: ModelEndpoint, n_calls: int, approved: bool) -> None:
    """Stop before spending money. Remote endpoints cost money, so they need --yes."""
    if is_local_url(endpoint.base_url) or approved:
        return
    raise UserError(
        f"{endpoint.base_url} is a remote endpoint and this would make {n_calls} paid calls. "
        "Check the cost, then re-run with --yes."
    )


def make_client(endpoint: ModelEndpoint, max_retries: int = 3, timeout: float = 300.0) -> OpenAI:
    """An OpenAI-style client for any endpoint. The key is read from .env, never from the YAML."""
    load_dotenv(paths.ENV_FILE)
    key = "not-needed"  # local servers ignore the key, but the client wants some text
    if endpoint.api_key_env:
        key = os.environ.get(endpoint.api_key_env, "")
        if not key:
            raise UserError(f"Set {endpoint.api_key_env} in .env first (see .env.example).")
    # max_retries: the SDK backs off and retries on rate limits (429) and server errors by itself.
    return OpenAI(base_url=endpoint.base_url, api_key=key, max_retries=max_retries, timeout=timeout)


def response_format_for(kind: str) -> dict[str, Any] | None:
    """How hard to push the model towards valid JSON.

    none         just the prompt
    json_object  the API promises some JSON
    json_schema  the API enforces our exact schema (constrained decoding)
    """
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "TicketTriage", "schema": TicketTriage.model_json_schema()},
        }
    return None
