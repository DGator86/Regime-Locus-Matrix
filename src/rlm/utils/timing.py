"""Stage timing utilities for RLM.

Provides ``timed_stage`` — a context manager that logs start/end/failure
with elapsed duration and structured context fields.

Usage::

    from rlm.utils.timing import timed_stage
    from rlm.utils.logging import get_logger

    log = get_logger(__name__)

    with timed_stage(log, "forecast", run_id="abc123", symbol="SPY"):
        result = pipeline.run(bars)
    # emits two log lines:
    #   stage=forecast run_id=abc123 symbol=SPY  start
    #   stage=forecast run_id=abc123 symbol=SPY  done  duration=1.23s
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator
import logging


@contextmanager
def timed_stage(
    logger: logging.Logger,
    stage: str,
    level: int = logging.INFO,
    **context: object,
) -> Generator[None, None, None]:
    """Context manager that logs stage start/done/failed with elapsed time.

    Parameters
    ----------
    logger:
        Logger instance.
    stage:
        Short stage label (e.g. ``"factors"``, ``"forecast"``, ``"backtest"``).
    level:
        Log level for start/done messages (default ``INFO``).
    **context:
        Arbitrary key=value pairs appended to every log line.

    On exception the context manager re-raises after logging the failure.
    """
    ctx_parts = [f"stage={stage}"] + [f"{k}={v}" for k, v in context.items()]
    prefix = "  ".join(ctx_parts)

    logger.log(level, "%s  start", prefix)
    t0 = time.monotonic()
    try:
        yield
        duration = time.monotonic() - t0
        logger.log(level, "%s  done  duration=%.2fs", prefix, duration)
    except Exception as exc:
        duration = time.monotonic() - t0
        logger.error("%s  failed  duration=%.2fs  error=%s", prefix, duration, exc)
        raise


def elapsed_since(t0: float) -> float:
    """Return seconds elapsed since monotonic timestamp *t0*."""
    return time.monotonic() - t0
