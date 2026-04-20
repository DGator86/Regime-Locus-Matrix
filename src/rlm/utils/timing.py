from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed_stage(logger, stage: str, **fields):
    start = time.monotonic()
    logger.info("stage start", extra={"stage": stage, **fields})
    try:
        yield
        logger.info("stage done", extra={"stage": stage, "success": True, "duration_s": round(time.monotonic() - start, 4), **fields})
    except Exception:
        logger.exception("stage failed", extra={"stage": stage, "success": False, "duration_s": round(time.monotonic() - start, 4), **fields})
        raise
