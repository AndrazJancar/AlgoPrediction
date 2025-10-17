from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # fallback if not installed
    load_dotenv = None  # type: ignore


def get_env(name: str) -> Optional[str]:
    """Load .env (if python-dotenv is installed) and return env var."""
    if load_dotenv is not None:
        load_dotenv()  # idempotent
    value = os.environ.get(name)
    if value is not None and value.strip() == "":
        return None
    return value


def get_entsoe_api_key() -> Optional[str]:
    return get_env("ENTSOE_API_KEY")


