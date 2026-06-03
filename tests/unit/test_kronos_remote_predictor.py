"""Remote GPU Kronos client wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from rlm.forecasting.kronos_config import KronosConfig
from rlm.forecasting.models.kronos.predictor import RLMKronosPredictor


def _bars(n: int = 40) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="D")
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def test_prepare_df_without_timestamp_column() -> None:
    df = _bars().drop(columns=["timestamp"])
    ohlcv, ts = RLMKronosPredictor._prepare_df(df)
    assert len(ohlcv) == len(df)
    assert len(ts) == len(df)


def test_remote_predict_paths_uses_env_url(monkeypatch) -> None:
    monkeypatch.setenv("RLM_KRONOS_REMOTE_URL", "http://gpu.test:8000")
    cfg = KronosConfig(pred_len=3, sample_count=2)
    pred = RLMKronosPredictor(cfg)

    fake_paths = np.ones((2, 3, 6), dtype=float)
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"paths": fake_paths.tolist(), "elapsed_ms": 12.0}

    with patch("requests.post", return_value=mock_resp) as post:
        out = pred.predict_paths(_bars())

    assert out.shape == (2, 3, 6)
    post.assert_called_once()
    url = post.call_args[0][0]
    assert url.endswith("/predict_paths")
    body = post.call_args[1]["json"]
    assert body["pred_len"] == 3
    assert len(body["bars"]["close"]) == 40
