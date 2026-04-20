"""Centralised logger factory and structured logging helpers for RLM.

Usage::

    from rlm.utils.logging import get_logger
    log = get_logger(__name__)

    log.info("forecast start  symbol=%s regime=%s", symbol, regime)

Environment variables:

    RLM_LOG_LEVEL   Log verbosity: DEBUG | INFO | WARNING | ERROR (default: INFO)
    RLM_LOG_JSON    Set to "1" for newline-delimited JSON output

Structured context helper::

    from rlm.utils.logging import log_stage

    with log_stage(log, "factors", symbol="SPY"):
        result = pipeline.run(bars)
    # emits:  stage=factors symbol=SPY ... done  duration=1.23s
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Generator


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger for *name*.

    Idempotent — safe to call at module level.  All RLM loggers share the same
    format/level so output is predictable across CLI, services, and pipelines.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = os.environ.get("RLM_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if os.environ.get("RLM_LOG_JSON", "").strip() == "1":
        import json as _json

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload: dict = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                # Carry through any extra context fields injected via LogRecord
                for key in ("command", "symbol", "stage", "duration_s"):
                    if hasattr(record, key):
                        payload[key] = getattr(record, key)
                return _json.dumps(payload)

        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    logger.addHandler(handler)
    logger.propagate = False
    return logger


@contextmanager
def log_stage(
    logger: logging.Logger,
    stage: str,
    **context: object,
) -> Generator[None, None, None]:
    """Context manager that logs stage start/end with elapsed duration.

    Parameters
    ----------
    logger:
        Logger instance (from ``get_logger``).
    stage:
        Short stage name, e.g. ``"factors"``, ``"forecast"``, ``"backtest"``.
    **context:
        Arbitrary key=value pairs appended to log lines (e.g. ``symbol="SPY"``).

    Example::

        with log_stage(log, "forecast", symbol="SPY", regime="hmm"):
            result = pipeline.run(bars)
        # logs:
        #   stage=forecast symbol=SPY regime=hmm  start
        #   stage=forecast symbol=SPY regime=hmm  done  duration=1.23s
    """
    ctx_str = "  ".join(f"{k}={v}" for k, v in context.items())
    prefix = f"stage={stage}  {ctx_str}" if ctx_str else f"stage={stage}"

    logger.info("%s  start", prefix)
    t0 = time.monotonic()
    try:
        yield
        duration = time.monotonic() - t0
        logger.info("%s  done  duration=%.2fs", prefix, duration)
    except Exception as exc:
        duration = time.monotonic() - t0
        logger.error("%s  failed  duration=%.2fs  error=%s", prefix, duration, exc)
        raise
