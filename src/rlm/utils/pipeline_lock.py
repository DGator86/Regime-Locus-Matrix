"""Single-instance lock for long-running universe pipeline jobs."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows dev
    fcntl = None  # type: ignore[assignment]


class PipelineLockError(RuntimeError):
    """Raised when the lock file cannot be acquired."""


@contextmanager
def universe_pipeline_lock(root: Path, *, name: str = "universe_pipeline") -> Iterator[None]:
    """Exclusive non-blocking lock under ``data/processed/.{name}.lock``.

    Raises ``PipelineLockError`` if another process holds the lock.
    """
    if fcntl is None:
        yield
        return

    lock_dir = root / "data" / "processed"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{name}.lock"
    fh = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineLockError(
                f"another {name} run is already in progress (lock: {lock_path})"
            ) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()}\n")
        fh.flush()
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


def universe_pipeline_lock_path(root: Path, *, name: str = "universe_pipeline") -> Path:
    """Lock file path under ``data/processed/.{name}.lock``."""
    return root / "data" / "processed" / f".{name}.lock"


def universe_pipeline_lock_age_sec(root: Path, *, name: str = "universe_pipeline") -> float | None:
    """Seconds since lock file mtime, or ``None`` if missing."""
    path = universe_pipeline_lock_path(root, name=name)
    if not path.is_file():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def universe_pipeline_lock_recent(
    root: Path,
    *,
    name: str = "universe_pipeline",
    max_age_sec: float | None = None,
) -> bool:
    """True when a lock file exists and is younger than ``max_age_sec`` (default: pipeline timeout env)."""
    age = universe_pipeline_lock_age_sec(root, name=name)
    if age is None:
        return False
    if max_age_sec is None:
        try:
            max_age_sec = float((os.environ.get("RLM_PIPELINE_TIMEOUT_SEC") or "2700").strip())
        except ValueError:
            max_age_sec = 2700.0
    return age <= max(60.0, float(max_age_sec))


def exit_if_universe_pipeline_busy(root: Path, *, name: str = "universe_pipeline") -> None:
    """Exit 0 when lock is held (skip duplicate run); re-raise on other errors."""
    try:
        with universe_pipeline_lock(root, name=name):
            return
    except PipelineLockError as exc:
        print(f"[pipeline-lock] skip — {exc}", flush=True)
        raise SystemExit(0) from exc
