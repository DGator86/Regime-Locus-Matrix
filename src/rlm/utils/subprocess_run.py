"""Subprocess helpers with optional timeout (used by orchestrators)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_with_timeout(
    cmd: list[str],
    *,
    cwd: Path | str,
    timeout_sec: float | None = None,
    env_key: str = "RLM_SUBPROCESS_TIMEOUT_SEC",
    default_timeout: float | None = None,
) -> int:
    """Run ``cmd``; kill the process group on timeout. Returns 124 on timeout."""
    if timeout_sec is None:
        raw = (os.environ.get(env_key) or "").strip()
        if raw:
            try:
                timeout_sec = float(raw)
            except ValueError:
                timeout_sec = default_timeout
        else:
            timeout_sec = default_timeout

    print("+", " ".join(cmd), flush=True)
    try:
        if timeout_sec is not None and timeout_sec > 0:
            proc = subprocess.run(cmd, cwd=str(cwd), timeout=timeout_sec)
            return int(proc.returncode)
        proc = subprocess.run(cmd, cwd=str(cwd))
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        print(
            f"[timeout] exceeded {timeout_sec:.0f}s — killed: {' '.join(cmd[:3])}…",
            flush=True,
        )
        return 124
