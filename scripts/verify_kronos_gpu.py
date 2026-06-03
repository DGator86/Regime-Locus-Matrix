#!/usr/bin/env python3
"""Verify RunPod/local Kronos GPU: GET /health and POST /predict_paths."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _bars(n: int = 48) -> pd.DataFrame:
    close = np.linspace(500.0, 502.0, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    base = (os.environ.get("RLM_KRONOS_REMOTE_URL") or "").strip().rstrip("/")
    if not base:
        print("verify_kronos_gpu: RLM_KRONOS_REMOTE_URL not set — skipping remote GPU probe")
        return 0

    timeout = float(os.environ.get("RLM_KRONOS_REMOTE_TIMEOUT_SEC", "120"))
    print(f"verify_kronos_gpu: health GET {base}/health")
    hr = requests.get(f"{base}/health", timeout=min(timeout, 30.0))
    hr.raise_for_status()
    print(f"  health_ok status={hr.status_code} body={hr.text[:200]}")

    from rlm.forecasting.kronos_config import KronosConfig
    from rlm.forecasting.models.kronos.predictor import RLMKronosPredictor

    pred = RLMKronosPredictor(KronosConfig(pred_len=2, sample_count=2, lookback=40))
    paths = pred.predict_paths(_bars(48))
    print(f"  predict_paths_ok shape={paths.shape} remote={pred._remote is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
