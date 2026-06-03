"""RLM-aware Kronos predictor — remote GPU service, local torch, or test delegate."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rlm.forecasting.kronos_config import KronosConfig

if TYPE_CHECKING:
    from rlm.forecasting.models.kronos.model.kronos import KronosPredictor

logger = logging.getLogger(__name__)

_PRICE_COLS = ["open", "high", "low", "close"]
_KRONOS_COLS = ["open", "high", "low", "close", "volume", "amount"]


class _RemoteKronosClient:
    """POST bar windows to a FastAPI Kronos GPU worker (``RLM_KRONOS_REMOTE_URL``)."""

    def __init__(self, base_url: str, config: KronosConfig) -> None:
        self._base = base_url.rstrip("/")
        self._config = config

    def predict_paths(
        self,
        df: pd.DataFrame,
        future_timestamps: Any | None = None,
    ) -> np.ndarray:
        import requests

        ohlcv, ts = RLMKronosPredictor._prepare_df(df)
        _ = future_timestamps  # server derives horizon from pred_len
        cfg = self._config
        payload = {
            "bars": {
                "open": ohlcv["open"].astype(float).tolist(),
                "high": ohlcv["high"].astype(float).tolist(),
                "low": ohlcv["low"].astype(float).tolist(),
                "close": ohlcv["close"].astype(float).tolist(),
                "volume": ohlcv["volume"].astype(float).tolist(),
                "amount": ohlcv["amount"].astype(float).tolist(),
                "timestamps": [str(t) for t in ts],
            },
            "pred_len": int(cfg.pred_len),
            "sample_count": int(cfg.sample_count),
            "temperature": float(cfg.temperature),
            "top_p": float(cfg.top_p),
            "top_k": 0,
        }
        timeout = float(os.environ.get("RLM_KRONOS_REMOTE_TIMEOUT_SEC", "120"))
        resp = requests.post(
            f"{self._base}/predict_paths",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        paths = np.asarray(data["paths"], dtype=float)
        if paths.ndim != 3:
            raise ValueError(f"remote predict_paths expected 3D array, got {paths.shape}")
        return paths


class RLMKronosPredictor:
    """Kronos paths for regime overlay: remote GPU (preferred on CPU VPS), local torch, or inject."""

    def __init__(
        self,
        config: KronosConfig | None = None,
        predictor: Any | None = None,
    ) -> None:
        self.config = config or KronosConfig.from_yaml()
        self._delegate = predictor
        self._predictor: KronosPredictor | None = None
        self._remote: _RemoteKronosClient | None = None

        self._cache_key: str | None = None
        self._cache_mean: np.ndarray | None = None
        self._cache_paths: np.ndarray | None = None
        self._cache_age: int = 0

        if predictor is not None:
            return

        remote = (os.environ.get("RLM_KRONOS_REMOTE_URL") or "").strip()
        if remote:
            self._remote = _RemoteKronosClient(remote, self.config)
            logger.info("Kronos: using remote GPU inference at %s", remote)
            return

        vendor = (os.environ.get("RLM_KRONOS_VENDOR_PATH") or "").strip()
        if vendor:
            vendor_path = Path(vendor).expanduser().resolve()
            if vendor_path.is_dir() and str(vendor_path) not in sys.path:
                sys.path.insert(0, str(vendor_path))

    def _ensure_loaded(self) -> KronosPredictor:
        if self._predictor is not None:
            return self._predictor

        from rlm.forecasting.models.kronos.model.kronos import Kronos, KronosPredictor, KronosTokenizer

        cfg = self.config
        device = (os.environ.get("RLM_KRONOS_DEVICE") or cfg.device or "cpu").strip()
        tok_path = cfg.finetuned_model_path or cfg.tokenizer_name
        model_path = cfg.finetuned_model_path or cfg.model_name

        if cfg.finetuned_model_path:
            logger.info("Loading finetuned Kronos from %s", cfg.finetuned_model_path)
        else:
            logger.info("Loading Kronos from HuggingFace: %s / %s", cfg.model_name, cfg.tokenizer_name)

        tokenizer = KronosTokenizer.from_pretrained(
            tok_path if cfg.finetuned_model_path else cfg.tokenizer_name
        )
        model = Kronos.from_pretrained(model_path if cfg.finetuned_model_path else cfg.model_name)
        self._predictor = KronosPredictor(
            model,
            tokenizer,
            device=device,
            max_context=cfg.max_context,
        )
        return self._predictor

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        out = df.copy()
        ts_col = "timestamp" if "timestamp" in out.columns else "timestamps"
        if ts_col not in out.columns and out.index.name in ("timestamp", "timestamps"):
            out = out.reset_index()
            ts_col = out.columns[0]
        if ts_col in out.columns:
            timestamps = pd.to_datetime(out[ts_col])
        elif isinstance(out.index, pd.DatetimeIndex):
            timestamps = pd.Series(out.index)
        else:
            n = len(out)
            timestamps = pd.date_range("2020-01-01", periods=n, freq="D")

        for col in _PRICE_COLS:
            if col not in out.columns:
                raise ValueError(f"Missing required column: {col}")

        if "volume" not in out.columns:
            out["volume"] = 0.0
        if "amount" not in out.columns:
            out["amount"] = out["volume"] * out["close"]

        return out[_KRONOS_COLS].copy(), timestamps

    @staticmethod
    def _cache_key_for(timestamps: pd.Series) -> str:
        raw = f"{timestamps.iloc[0]}_{timestamps.iloc[-1]}_{len(timestamps)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def predict_paths(
        self,
        df: pd.DataFrame,
        future_timestamps: Any | None = None,
    ) -> np.ndarray:
        if self._delegate is not None:
            try:
                return np.asarray(
                    self._delegate.predict_paths(df, future_timestamps=future_timestamps),
                    dtype=float,
                )
            except TypeError:
                return np.asarray(self._delegate.predict_paths(df), dtype=float)

        cfg = self.config
        ohlcv, ts = self._prepare_df(df)
        cache_key = self._cache_key_for(ts)
        if (
            self._cache_paths is not None
            and self._cache_key == cache_key
            and self._cache_age < cfg.cache_ttl_bars
        ):
            self._cache_age += 1
            return self._cache_paths

        if self._remote is not None:
            paths = self._remote.predict_paths(df, future_timestamps=future_timestamps)
        else:
            if future_timestamps is None:
                freq = ts.diff().median()
                future_timestamps = pd.Series([ts.iloc[-1] + freq * (i + 1) for i in range(cfg.pred_len)])
            future_timestamps = pd.to_datetime(future_timestamps)
            predictor = self._ensure_loaded()
            paths = np.asarray(
                predictor.predict(
                    df=ohlcv,
                    x_timestamp=ts,
                    y_timestamp=future_timestamps,
                    pred_len=cfg.pred_len,
                    T=cfg.temperature,
                    top_k=0,
                    top_p=cfg.top_p,
                    sample_count=cfg.sample_count,
                    verbose=False,
                    return_all_paths=True,
                ),
                dtype=float,
            )

        self._cache_key = cache_key
        self._cache_paths = paths
        self._cache_mean = np.mean(paths, axis=0)
        self._cache_age = 0
        return paths

    def predict_mean(
        self,
        df: pd.DataFrame,
        future_timestamps: Any | None = None,
    ) -> np.ndarray:
        if self._cache_mean is not None and self._cache_age < self.config.cache_ttl_bars:
            return self._cache_mean
        self.predict_paths(df, future_timestamps)
        assert self._cache_mean is not None
        return self._cache_mean

    def invalidate_cache(self) -> None:
        self._cache_key = None
        self._cache_mean = None
        self._cache_paths = None
        self._cache_age = 0
