"""
KronosForecastPipeline — integrates the Kronos foundation model as a forecast
source inside the Regime-Locus-Matrix forecasting stack.

Kronos (https://github.com/DGator86/Kronos) is a decoder-only Transformer
pre-trained on K-line (OHLCV) data from 45+ global exchanges.  It predicts
future OHLCV bars autoregressively.

This pipeline wraps ``KronosPredictor`` so its predicted close prices are
converted into return forecasts compatible with the RLM ROEE policy engine.
The base RLM distributional layer (regime scores, bands) is always computed
first; Kronos overrides the ``mu / sigma / forecast_return_*`` columns.

Usage
-----
::

    from rlm.forecasting.kronos_forecast import KronosConfig, KronosForecastPipeline

    cfg = KronosConfig(
        model_name="NeoQuasar/Kronos-small",
        tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
        sample_count=5,   # MC samples → quantile bounds
        stride=1,         # inference every bar (most accurate)
    )
    pipe = KronosForecastPipeline(config=cfg)
    df_out = pipe.run(df_ohlcv)

Install the required extras before use::

    pip install -e ".[kronos]"
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from rlm.forecasting.bands import compute_state_matrix_bands
from rlm.forecasting.distribution import estimate_distribution
from rlm.forecasting.kronos_config import KronosConfig
from rlm.types.forecast import ForecastConfig

if TYPE_CHECKING:
    from rlm.forecasting.models.kronos.model.kronos import KronosPredictor


def _make_future_timestamps(index: pd.Index, n: int) -> pd.DatetimeIndex:
    """Generate *n* plausible future timestamps following *index*."""
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        delta = index[-1] - index[-2]
        start = index[-1] + delta
        try:
            freq = pd.tseries.frequencies.to_offset(delta)
        except Exception:
            freq = None
        if freq is not None:
            return pd.date_range(start=start, periods=n, freq=freq)
        return pd.DatetimeIndex([start + i * delta for i in range(n)])
    # Fallback: synthetic business-day timestamps
    return pd.date_range(start=pd.Timestamp("2000-01-01"), periods=n, freq="B")


class KronosForecastPipeline:
    """Wraps the Kronos foundation model as an RLM forecast source.

    For every bar in the input DataFrame, the pipeline takes the preceding
    ``lookback`` bars as Kronos context and generates a one-step-ahead
    close-price forecast.  That predicted close is converted to a return
    estimate which replaces the RLM distributional ``mu / sigma /
    forecast_return_*`` columns.

    When ``sample_count > 1`` the pipeline draws that many independent
    stochastic samples from the Kronos decoder and uses the empirical
    distribution as forecast-return quantiles.  When ``sample_count == 1``
    it falls back to a realised-vol interval around the point estimate.

    Inference is gated: any ``ImportError`` (torch / Kronos not installed) or
    runtime error degrades gracefully to the standard distributional fallback.

    Parameters
    ----------
    config:
        Kronos-specific settings; defaults to ``KronosConfig()``.
    rlm_config:
        Base RLM distributional config (sigma_floor, quantile bounds, …).
    move_window:
        Rolling window passed to the base ``estimate_distribution`` call.
    vol_window:
        Realised-vol rolling window.
    """

    def __init__(
        self,
        config: KronosConfig | None = None,
        rlm_config: ForecastConfig | None = None,
        move_window: int = 100,
        vol_window: int = 100,
        predictor: Any | None = None,
        min_lookback: int = 30,
    ) -> None:
        self.config = config or KronosConfig()
        self.rlm_config = rlm_config or ForecastConfig()
        self.move_window = move_window
        self.vol_window = vol_window
        self._predictor: KronosPredictor | None = None
        self._predictor_override: Any | None = predictor
        self.min_lookback = int(min_lookback)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_predictor(self) -> "KronosPredictor":
        if self._predictor_override is not None:
            return self._predictor_override  # type: ignore[return-value]
        if self._predictor is not None:
            return self._predictor
        try:
            from rlm.forecasting.models.kronos.model.kronos import (
                Kronos,
                KronosPredictor,
                KronosTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "Kronos dependencies not found. Install with: pip install -e '.[kronos]'\n"
                "Required: torch>=2.0, einops, huggingface-hub, safetensors, tqdm."
            ) from exc

        tokenizer = KronosTokenizer.from_pretrained(self.config.tokenizer_name)
        model = Kronos.from_pretrained(self.config.model_name)
        self._predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=self.config.device,
            max_context=self.config.max_context,
            clip=self.config.clip,
        )
        return self._predictor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the base RLM layer, then overlay Kronos return forecasts.

        Parameters
        ----------
        df:
            Input bar DataFrame with at minimum ``open, high, low, close``
            columns (``volume`` is optional; will be filled with zeros if
            absent).

        Returns
        -------
        pd.DataFrame
            Same index as *df*.  Kronos-specific columns added:

            * ``forecast_return``, ``forecast_return_median``,
              ``forecast_return_lower``, ``forecast_return_upper``
            * ``forecast_uncertainty``
            * ``mu``, ``sigma``, ``mean_price``
            * ``forecast_source`` == ``"kronos"``

            All standard RLM regime/scoring/band columns are also present.
            On Kronos failure, ``forecast_source`` == ``"distribution_fallback"``.
        """
        base = estimate_distribution(
            df=df,
            config=self.rlm_config,
            move_window=self.move_window,
            vol_window=self.vol_window,
        )
        out = compute_state_matrix_bands(base)

        try:
            kronos_df = self._run_kronos_rolling(df)
        except Exception as exc:
            warnings.warn(
                f"KronosForecastPipeline: Kronos inference failed, "
                f"using distribution_fallback. Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            out["forecast_source"] = "distribution_fallback"
            return out

        cfg = self.config
        out["forecast_return_median"] = kronos_df["median"].values
        out["forecast_return_lower"] = kronos_df["lower"].values
        out["forecast_return_upper"] = kronos_df["upper"].values
        out["forecast_return"] = out["forecast_return_median"]
        out["forecast_uncertainty"] = (
            (out["forecast_return_upper"] - out["forecast_return_lower"]).clip(lower=0.0)
        )
        out["mu"] = out["forecast_return_median"]

        z_span = float(norm.ppf(cfg.upper_quantile) - norm.ppf(cfg.lower_quantile))
        if abs(z_span) > 1e-9:
            sigma_series = (
                (out["forecast_return_upper"] - out["forecast_return_lower"]).abs() / z_span
            )
        else:
            sigma_series = pd.Series(cfg.sigma_floor, index=out.index)
        out["sigma"] = sigma_series.clip(lower=cfg.sigma_floor)
        out["mean_price"] = out["close"] * (1.0 + out["forecast_return"])
        out["forecast_source"] = "kronos"
        return out

    # ------------------------------------------------------------------
    # Rolling inference
    # ------------------------------------------------------------------

    def _run_kronos_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run stride-based rolling Kronos inference across the full bar series.

        Returns a DataFrame with columns ``median``, ``lower``, ``upper``
        aligned to *df*'s index.  Values for the warm-up period (first
        ``min_ctx`` bars) are back-filled from the first available inference.
        """
        predictor = self._get_predictor()
        if self._predictor_override is not None:
            return self._run_kronos_rolling_predict_paths(df, predictor)
        cfg = self.config
        n = len(df)
        lookback = min(cfg.lookback, cfg.max_context)
        stride = max(1, cfg.stride)
        min_ctx = max(2, cfg.pred_len)

        # Synthesise DatetimeIndex if the df uses an integer index
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        else:
            timestamps = pd.date_range(start="2000-01-01", periods=n, freq="B")

        medians = np.full(n, np.nan)
        lowers = np.full(n, np.nan)
        uppers = np.full(n, np.nan)

        for i in range(min_ctx - 1, n, stride):
            ctx_start = max(0, i - lookback + 1)
            x_df = df.iloc[ctx_start : i + 1][["open", "high", "low", "close"]].copy()
            x_ts = timestamps[ctx_start : i + 1]
            y_ts = _make_future_timestamps(x_ts, cfg.pred_len)

            current_close = float(df["close"].iloc[i])
            if current_close == 0.0:
                continue

            med, lo, hi = self._single_forecast(predictor, x_df, x_ts, y_ts, current_close, df, ctx_start, i)

            # Fill this result forward across the stride window
            end_j = min(i + stride, n)
            medians[i:end_j] = med
            lowers[i:end_j] = lo
            uppers[i:end_j] = hi

        # Back-fill warm-up rows from the first valid inference
        first_valid = int(np.argmax(~np.isnan(medians))) if not np.all(np.isnan(medians)) else -1
        if first_valid > 0:
            medians[:first_valid] = medians[first_valid]
            lowers[:first_valid] = lowers[first_valid]
            uppers[:first_valid] = uppers[first_valid]

        return pd.DataFrame(
            {"median": medians, "lower": lowers, "upper": uppers},
            index=df.index,
        )

    def _run_kronos_rolling_predict_paths(self, df: pd.DataFrame, pred: Any) -> pd.DataFrame:
        """Rolling forecast using a ``predict_paths(df[, future_timestamps]) -> (S,T,6)`` API (tests / mocks)."""
        cfg = self.config
        n = len(df)
        min_lb = max(self.min_lookback, cfg.pred_len + 1, 2)
        medians = np.full(n, np.nan)
        lowers = np.full(n, np.nan)
        uppers = np.full(n, np.nan)

        for i in range(min_lb, n):
            lookback = min(cfg.lookback, cfg.max_context)
            ctx_start = max(0, i - lookback + 1)
            ctx = df.iloc[ctx_start : i + 1]
            current_close = float(df["close"].iloc[i])
            if current_close == 0.0:
                continue
            try:
                paths = pred.predict_paths(ctx, future_timestamps=None)
            except TypeError:
                paths = pred.predict_paths(ctx)
            arr = np.asarray(paths, dtype=float)
            if arr.ndim != 3 or arr.shape[0] < 1:
                continue
            sample_returns = (arr[:, -1, 3] - current_close) / current_close
            med = float(np.median(sample_returns))
            lo = float(np.percentile(sample_returns, cfg.lower_quantile * 100))
            hi = float(np.percentile(sample_returns, cfg.upper_quantile * 100))
            medians[i] = med
            lowers[i] = lo
            uppers[i] = hi

        first_valid = int(np.argmax(~np.isnan(medians))) if not np.all(np.isnan(medians)) else -1
        if first_valid > 0:
            medians[:first_valid] = medians[first_valid]
            lowers[:first_valid] = lowers[first_valid]
            uppers[:first_valid] = uppers[first_valid]

        return pd.DataFrame(
            {"median": medians, "lower": lowers, "upper": uppers},
            index=df.index,
        )

    def _single_forecast(
        self,
        predictor: "KronosPredictor",
        x_df: pd.DataFrame,
        x_ts: pd.DatetimeIndex,
        y_ts: pd.DatetimeIndex,
        current_close: float,
        full_df: pd.DataFrame,
        ctx_start: int,
        ctx_end_idx: int,
    ) -> tuple[float, float, float]:
        """Run one Kronos inference and return (median, lower, upper) returns.

        When ``sample_count > 1``, draws that many independent samples and
        computes empirical quantiles.  Otherwise uses a realised-vol interval.
        """
        cfg = self.config

        if cfg.sample_count > 1:
            sample_returns = []
            for _ in range(cfg.sample_count):
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=cfg.pred_len,
                    T=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    sample_count=1,
                    verbose=False,
                )
                pred_close = float(pred_df["close"].iloc[0])
                sample_returns.append((pred_close - current_close) / current_close)

            arr = np.array(sample_returns)
            med = float(np.median(arr))
            lo = float(np.percentile(arr, cfg.lower_quantile * 100))
            hi = float(np.percentile(arr, cfg.upper_quantile * 100))
        else:
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=cfg.pred_len,
                T=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                sample_count=1,
                verbose=cfg.verbose,
            )
            pred_close = float(pred_df["close"].iloc[0])
            med = (pred_close - current_close) / current_close

            # Fallback interval: ±1σ of recent realised returns
            hist_rets = full_df["close"].pct_change().iloc[ctx_start : ctx_end_idx + 1].dropna()
            rv = float(hist_rets.std()) if len(hist_rets) > 1 else cfg.sigma_floor
            lo = med - rv
            hi = med + rv

        return med, lo, hi

    def compute_kronos_overlay(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run only the Kronos inference layer (no base RLM distributional pass).

        Useful for ``apply_kronos_blend``, where the base distributional layer
        has already been run by a different pipeline.

        Returns
        -------
        pd.DataFrame
            Columns ``median``, ``lower``, ``upper`` aligned to *df*'s index.
        """
        return self._run_kronos_rolling(df)


# ---------------------------------------------------------------------------
# Composable blend helpers
# ---------------------------------------------------------------------------

def apply_kronos_blend(
    forecast_df: pd.DataFrame,
    config: KronosConfig | None = None,
    weight: float = 0.35,
) -> pd.DataFrame:
    """Blend Kronos return forecasts into an existing forecast DataFrame.

    Designed to be called *after* any RLM pipeline has already run
    (``ForecastPipeline``, ``HybridForecastPipeline``, etc.).  Kronos adds a
    deep-learning signal that is linearly interpolated with the distributional
    model at the specified ``weight``:

    .. code-block:: text

        mu_out         = (1 - weight) * mu_base         + weight * mu_kronos
        sigma_out      = (1 - weight) * sigma_base      + weight * sigma_kronos
        forecast_return_median_out = ...  (same blend)

    The Kronos model is loaded lazily on the first call and cached for the
    lifetime of the process.  If Kronos is unavailable (torch not installed,
    network error, …) the function returns *forecast_df* unmodified and emits
    a ``RuntimeWarning``.

    Parameters
    ----------
    forecast_df:
        Output of any RLM forecast pipeline.  Must contain at minimum
        ``open, high, low, close`` (for Kronos context) and the standard
        distributional columns ``mu``, ``sigma``, ``forecast_return``.
    config:
        Kronos inference settings.  Defaults to ``KronosConfig()``.
    weight:
        Blend weight for the Kronos signal (0.0 → keep base only,
        1.0 → Kronos only, default ``0.35``).

    Returns
    -------
    pd.DataFrame
        Same index/shape as *forecast_df* with blended forecast columns and
        ``forecast_source`` updated to reflect the blend (e.g.
        ``"kronos_blend_0.35_over_hmm"``).
    """
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"weight must be in [0, 1], got {weight!r}.")

    if weight == 0.0:
        return forecast_df  # no-op

    cfg = config or KronosConfig()
    pipe = KronosForecastPipeline(config=cfg)

    try:
        kronos_df = pipe.compute_kronos_overlay(forecast_df)
    except Exception as exc:
        warnings.warn(
            f"apply_kronos_blend: Kronos inference failed; returning base forecast unchanged. "
            f"Reason: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return forecast_df

    out = forecast_df.copy()
    w = weight
    base_source = str(out.get("forecast_source", pd.Series(["unknown"])).iloc[-1])

    def _blend(base_col: str, kronos_col: str) -> pd.Series:
        base = pd.to_numeric(out.get(base_col, pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        kron = kronos_df[kronos_col].fillna(base)
        return (1.0 - w) * base + w * kron

    out["forecast_return_median"] = _blend("forecast_return_median", "median")
    out["forecast_return_lower"] = _blend("forecast_return_lower", "lower")
    out["forecast_return_upper"] = _blend("forecast_return_upper", "upper")
    out["forecast_return"] = out["forecast_return_median"]
    out["forecast_uncertainty"] = (
        (out["forecast_return_upper"] - out["forecast_return_lower"]).clip(lower=0.0)
    )
    out["mu"] = out["forecast_return_median"]

    # Recompute sigma from blended interval
    z_span = float(norm.ppf(cfg.upper_quantile) - norm.ppf(cfg.lower_quantile))
    if abs(z_span) > 1e-9:
        sigma_series = (
            (out["forecast_return_upper"] - out["forecast_return_lower"]).abs() / z_span
        )
    else:
        sigma_series = pd.Series(cfg.sigma_floor, index=out.index)
    out["sigma"] = sigma_series.clip(lower=cfg.sigma_floor)
    out["mean_price"] = out["close"] * (1.0 + out["forecast_return"])
    out["forecast_source"] = f"kronos_blend_{w:.2f}_over_{base_source}"
    out["kronos_forecast_return"] = kronos_df["median"].values
    out["kronos_forecast_lower"] = kronos_df["lower"].values
    out["kronos_forecast_upper"] = kronos_df["upper"].values
    return out


class KronosBlendPipeline:
    """Wraps any existing RLM pipeline and applies a Kronos blend post-step.

    This is the preferred way to add Kronos to the production suite without
    touching existing pipeline code.  It is fully transparent: the ``run()``
    signature matches all existing pipeline variants and the output DataFrame
    is identical in structure to what the underlying pipeline produces, with
    the ``mu / sigma / forecast_return_*`` columns blended with Kronos.

    Usage (standalone)::

        from rlm.forecasting.kronos_forecast import KronosBlendPipeline, KronosConfig
        from rlm.forecasting.engines import HybridForecastPipeline

        base = HybridForecastPipeline(hmm_config=HMMConfig(n_states=6))
        pipeline = KronosBlendPipeline(base, weight=0.35)
        df_out = pipeline.run(df_features)

    Usage (via live model config — see ``LiveRegimeModelConfig.build_pipeline()``)::

        # live_model_config.json: { "use_kronos": true, "kronos": { "weight": 0.4 } }
        pipeline = live_model.build_pipeline()
        df_out = pipeline.run(feats)

    Parameters
    ----------
    base_pipeline:
        Any RLM pipeline instance with a ``.run(df, **kw)`` method.
    kronos_config:
        Kronos inference settings.
    weight:
        Blend weight for Kronos (default ``0.35``).
    """

    def __init__(
        self,
        base_pipeline: object,
        kronos_config: KronosConfig | None = None,
        weight: float = 0.35,
    ) -> None:
        self.base_pipeline = base_pipeline
        self.kronos_config = kronos_config or KronosConfig()
        self.weight = weight

    def run(self, df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Run the base pipeline then blend in Kronos forecasts."""
        import inspect
        sig = inspect.signature(self.base_pipeline.run)
        if "train_mask" in sig.parameters:
            base_out = self.base_pipeline.run(df, **kwargs)
        else:
            base_out = self.base_pipeline.run(df)
        return apply_kronos_blend(base_out, config=self.kronos_config, weight=self.weight)
