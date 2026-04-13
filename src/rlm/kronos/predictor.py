"""RLM-aware wrapper around KronosPredictor.

Handles bar-format adaptation, lazy model loading, prediction caching,
and multi-sample path support for regime confidence / probabilistic forecasting.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rlm.kronos.config import KronosConfig

if TYPE_CHECKING:
    from rlm.kronos.model.kronos import KronosPredictor

logger = logging.getLogger(__name__)

_PRICE_COLS = ["open", "high", "low", "close"]
_KRONOS_COLS = ["open", "high", "low", "close", "volume", "amount"]


class RLMKronosPredictor:
    """Thin adapter between RLM bar DataFrames and the vendored KronosPredictor.

    Features:
    * Lazy model loading (first call triggers HuggingFace download).
    * Normalises RLM column names (``timestamp`` -> ``timestamps``, fills
      missing ``amount`` from ``volume * close``).
    * Configurable prediction caching so backtest bar-loops do not
      re-run inference when the lookback shifts by one bar.
    * ``predict_paths`` returns all *sample_count* independent sample
      trajectories as an ndarray for downstream regime classification.
    """

    def __init__(self, config: KronosConfig | None = None) -> None:
        """
        Initialize the RLM-aware Kronos predictor and prepare lazy-loading and caching state.
        
        Parameters:
            config (KronosConfig | None): Optional configuration for the underlying Kronos model/tokenizer.
                If `None`, a default configuration is loaded via `KronosConfig.from_yaml()`.
        
        Description:
            Sets up internal state for lazy model/tokenizer loading and in-memory caching:
            - `_predictor` is left unset for lazy initialization.
            - Cache fields `_cache_key`, `_cache_mean`, `_cache_paths` are initialized to `None`.
            - `_cache_age` is initialized to 0.
        """
        self._config = config or KronosConfig.from_yaml()
        self._predictor: KronosPredictor | None = None

        self._cache_key: str | None = None
        self._cache_mean: np.ndarray | None = None
        self._cache_paths: np.ndarray | None = None
        self._cache_age: int = 0

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> KronosPredictor:
        """
        Lazily load and return the KronosPredictor instance configured for this wrapper.
        
        When a predictor is not already loaded, this method loads a tokenizer and model (preferring a configured finetuned model path when present), constructs a KronosPredictor using the wrapper's configuration (device and max_context), stores it on self._predictor, and returns it.
        
        Returns:
            KronosPredictor: The cached or newly created KronosPredictor instance.
        """
        if self._predictor is not None:
            return self._predictor

        from rlm.kronos.model.kronos import Kronos, KronosPredictor, KronosTokenizer

        cfg = self._config
        tok_path = cfg.finetuned_model_path or cfg.tokenizer_name
        model_path = cfg.finetuned_model_path or cfg.model_name

        if cfg.finetuned_model_path:
            logger.info("Loading finetuned Kronos from %s", cfg.finetuned_model_path)
        else:
            logger.info("Loading Kronos from HuggingFace: %s / %s", cfg.model_name, cfg.tokenizer_name)

        tokenizer = KronosTokenizer.from_pretrained(
            tok_path if cfg.finetuned_model_path else cfg.tokenizer_name
        )
        model = Kronos.from_pretrained(
            model_path if cfg.finetuned_model_path else cfg.model_name
        )
        self._predictor = KronosPredictor(
            model, tokenizer, device=cfg.device, max_context=cfg.max_context
        )
        return self._predictor

    # ------------------------------------------------------------------
    # Bar-format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Normalize an RLM bars DataFrame into the Kronos predictor input format.
        
        Accepts a DataFrame with OHLC bars where the timestamp may be in a column named "timestamp" or "timestamps" or as the index named "timestamp" / "timestamps". Validates presence of the price columns "open", "high", "low", and "close"; if "volume" is missing it is set to 0.0; if "amount" is missing it is derived as volume * close.
        
        Parameters:
            df (pd.DataFrame): Input bars DataFrame.
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple where the first element is an OHLCV+amount DataFrame with columns ["open", "high", "low", "close", "volume", "amount"] suitable for Kronos, and the second element is a pd.Series of timestamps corresponding to each row.
        """
        out = df.copy()

        ts_col = "timestamp" if "timestamp" in out.columns else "timestamps"
        if ts_col not in out.columns and out.index.name in ("timestamp", "timestamps"):
            out = out.reset_index()
            ts_col = out.columns[0]
        timestamps = pd.to_datetime(out[ts_col])

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
        """
        Compute a deterministic cache key for a series of timestamps.
        
        Parameters:
            timestamps (pd.Series): Ordered sequence of timestamps (first and last entries and the count are used).
        
        Returns:
            str: MD5 hex digest derived from the first timestamp, last timestamp, and the length of the series.
        """
        raw = f"{timestamps.iloc[0]}_{timestamps.iloc[-1]}_{len(timestamps)}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_paths(
        self,
        df: pd.DataFrame,
        future_timestamps: pd.Series | None = None,
    ) -> np.ndarray:
        """
        Generate multiple independent future OHLCV sample paths from historical bar data.
        
        The input DataFrame must contain columns "open", "high", "low", and "close"; "volume" will default to 0.0 if missing and "amount" will be derived as volume * close if missing. If `future_timestamps` is None, future timestamps are synthesized by extending the last observed interval (median delta) for `pred_len` steps. Results are cached and may be returned from an internal cache keyed by the input timestamps; a fresh prediction updates the cache.
        
        Parameters:
            df (pd.DataFrame): Historical bar data with timestamp information (column "timestamp" or "timestamps", or an index named "timestamp"/"timestamps") and OHLC(+V/A) columns.
            future_timestamps (pd.Series | None): Datetime series specifying prediction timestamps; if None, synthetic timestamps are generated.
        
        Returns:
            np.ndarray: Array of shape (sample_count, pred_len, 6) containing predicted paths. The final dimension columns are ordered as `open, high, low, close, volume, amount`.
        """
        cfg = self._config
        ohlcv, ts = self._prepare_df(df)

        cache_key = self._cache_key_for(ts)
        if (
            self._cache_paths is not None
            and self._cache_key == cache_key
            and self._cache_age < cfg.cache_ttl_bars
        ):
            self._cache_age += 1
            return self._cache_paths

        if future_timestamps is None:
            freq = ts.diff().median()
            future_timestamps = pd.Series(
                [ts.iloc[-1] + freq * (i + 1) for i in range(cfg.pred_len)]
            )
        future_timestamps = pd.to_datetime(future_timestamps)

        predictor = self._ensure_loaded()
        paths: np.ndarray = predictor.predict(
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
        )

        self._cache_key = cache_key
        self._cache_paths = paths
        self._cache_mean = np.mean(paths, axis=0)
        self._cache_age = 0
        return paths

    def predict_mean(
        self,
        df: pd.DataFrame,
        future_timestamps: pd.Series | None = None,
    ) -> np.ndarray:
        """
        Compute the mean OHLCV forecast for the prediction horizon.
        
        Parameters:
            df (pd.DataFrame): Input bar DataFrame containing OHLC and timestamp information.
            future_timestamps (pd.Series | None): Optional future timestamps to predict. If None, future timestamps are inferred by extending the input timestamps using their median interval.
        
        Returns:
            np.ndarray: Array of shape (pred_len, 6) containing the mean forecasted columns in order: open, high, low, close, volume, amount.
        """
        if (
            self._cache_mean is not None
            and self._cache_age < self._config.cache_ttl_bars
        ):
            return self._cache_mean
        self.predict_paths(df, future_timestamps)
        assert self._cache_mean is not None
        return self._cache_mean

    def invalidate_cache(self) -> None:
        """
        Clear all stored prediction cache entries and reset cache age.
        
        This invalidates the cached key, mean forecast, and sampled paths so subsequent
        predictions will trigger fresh inference; also resets the cache age counter to 0.
        """
        self._cache_key = None
        self._cache_mean = None
        self._cache_paths = None
        self._cache_age = 0
