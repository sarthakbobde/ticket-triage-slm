"""Talking to model servers: building the client and choosing how strictly to ask for JSON."""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from triage import paths
from triage.errors import UserError
from triage.schema import TicketTriage

log = logging.getLogger(__name__)


def is_local_url(base_url):
    """True if the server runs on this computer (so calls to it are free)."""
    if "localhost" in base_url:
        return True
    if "127.0.0.1" in base_url:
        return True
    return False


def require_approval_for_paid_calls(endpoint, n_calls, approved):
    """Stop before spending money. A remote server costs money, so it needs --yes."""
    if is_local_url(endpoint.base_url):
        return
    if approved:
        return
    raise UserError(
        endpoint.base_url
        + " is a remote server and this would make "
        + str(n_calls)
        + " paid calls. Check the cost, then run again with --yes."
    )


def make_client(endpoint, max_retries=3, timeout=300.0):
    """Build an OpenAI-style client for any server. The key comes from .env, never from the YAML file."""
    load_dotenv(paths.ENV_FILE)

    # Local servers ignore the key, but the client insists on some text.
    key = "not-needed"
    if endpoint.api_key_env:
        key = os.environ.get(endpoint.api_key_env, "")
        if key == "":
            raise UserError("Set " + endpoint.api_key_env + " in .env first (see .env.example).")

    # max_retries: the client waits and tries again by itself when the server says "too many
    # requests" (429) or has a temporary error.
    return OpenAI(base_url=endpoint.base_url, api_key=key, max_retries=max_retries, timeout=timeout)


def response_format_for(kind):
    """How hard to push the model towards valid JSON.

    none         just the prompt (returns None: nothing extra is sent)
    json_object  the server promises to return some JSON
    json_schema  the server enforces our exact schema (called "constrained decoding")
    """
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        schema_info = {"name": "TicketTriage", "schema": TicketTriage.model_json_schema()}
        return {"type": "json_schema", "json_schema": schema_info}
    return None
