"""Runtime updates to Dhan API credentials.

Persists to .env, updates os.environ, and mutates the (frozen) SETTINGS
instance in place so modules that already imported SETTINGS see the change.
"""
from __future__ import annotations

import os

from dotenv import set_key

from quant_intelligence.config.settings import PROJECT_ROOT, SETTINGS

ENV_PATH = PROJECT_ROOT / ".env"


def mask_secret(value: str, visible: int = 4) -> str:
    if not value:
        return "(not set)"
    if len(value) <= visible:
        return "*" * len(value)
    return "*" * 8 + value[-visible:]


def update_dhan_credentials(client_id: str, access_token: str) -> None:
    client_id, access_token = client_id.strip(), access_token.strip()
    if not client_id or not access_token:
        raise ValueError("Both Client ID and Access Token are required.")

    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), "DHAN_CLIENT_ID", client_id, quote_mode="never")
    set_key(str(ENV_PATH), "DHAN_ACCESS_TOKEN", access_token, quote_mode="never")

    os.environ["DHAN_CLIENT_ID"] = client_id
    os.environ["DHAN_ACCESS_TOKEN"] = access_token
    object.__setattr__(SETTINGS, "dhan_client_id", client_id)
    object.__setattr__(SETTINGS, "dhan_access_token", access_token)
