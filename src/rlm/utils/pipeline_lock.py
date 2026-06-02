"""Single-instance lock for long-running universe pipeline jobs."""

from __future__ import annotations

import os
import sys
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


def exit_if_universe_pipeline_busy(root: Path, *, name: str = "universe_pipeline") -> None:
    """Exit 0 when lock is held (skip duplicate run); re-raise on other errors."""
    try:
        with universe_pipeline_lock(root, name=name):
            return
    except PipelineLockError as exc:
        print(f"[pipeline-lock] skip — {exc}", flush=True)
        raise SystemExit(0) from exc
