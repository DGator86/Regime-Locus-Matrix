"""Centralised logger factory for RLM.

Usage::

    from rlm.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("Pipeline started: symbol=%s", symbol)
"""

from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger for *name*.

    Respects the ``RLM_LOG_LEVEL`` environment variable (default INFO).
    Set ``RLM_LOG_JSON=1`` for newline-delimited JSON output.
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
                return _json.dumps({
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                })

        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s", datefmt="%H:%M:%S")
        )

    logger.addHandler(handler)
    logger.propagate = False
    return logger
