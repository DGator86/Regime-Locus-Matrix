"""Atomic filesystem helpers for shared live artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Serialize ``payload`` to ``path`` via temp file + ``os.replace``.

    Concurrent readers never observe a truncated intermediate file from this writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
