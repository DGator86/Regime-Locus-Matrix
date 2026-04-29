from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, MutableMapping


class _ContextAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        extra: dict[str, Any] = dict(self.extra)
        nested = kwargs.pop("extra", None)
        if isinstance(nested, dict):
            extra.update(nested)
        kwargs["extra"] = extra
        return msg, kwargs


def _configure_logger(logger: logging.Logger) -> logging.Logger:
    if logger.handlers:
        return logger
    level = getattr(logging, os.environ.get("RLM_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if os.environ.get("RLM_LOG_JSON", "").strip() == "1":

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload: dict[str, Any] = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                for key in (
                    "run_id",
                    "command",
                    "symbol",
                    "backend",
                    "profile",
                    "stage",
                    "success",
                    "duration_s",
                ):
                    if hasattr(record, key):
                        payload[key] = getattr(record, key)
                return json.dumps(payload)

        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s", "%H:%M:%S"))

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return _configure_logger(logging.getLogger(name))


def get_run_logger(name: str, run_id: str | None = None, **context) -> logging.LoggerAdapter:
    logger = get_logger(name)
    extra = {"run_id": run_id, **context}
    return _ContextAdapter(logger, extra)
