from __future__ import annotations

import secrets
from datetime import datetime, timezone


def generate_run_id(prefix: str | None = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    return f"{prefix + '-' if prefix else ''}{ts}-{suffix}"
