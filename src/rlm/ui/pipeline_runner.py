"""Shared bars → factors → state matrix → forecast stack for Streamlit apps."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.live_model import LiveRegimeModelConfig
from rlm.forecasting.pipeline import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridMarkovForecastPipeline,
)
from rlm.forecasting.probabilistic import ProbabilisticForecastPipeline
from rlm.scoring.state_matrix import classify_state_matrix

ForecastMode = Literal["deterministic", "hmm", "markov", "probabilistic"]


def run_feature_forecast_stack(
    bars: pd.DataFrame,
    *,
    symbol: str,
    attach_vix: bool,
    move_window: int,
    vol_window: int,
    forecast_mode: ForecastMode,
    live: LiveRegimeModelConfig | None = None,
    probabilistic_model_path: str | Path | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Prepare enriched feature/state data from input market bars and run the selected forecasting pipeline, returning either the forecast dataframe or an error message.
    
    This function validates and enriches the provided bars, generates factor features and a classified state matrix, selects a forecast backend according to `forecast_mode`, runs that pipeline, appends a boolean `has_major_event` column (always False), and returns the resulting dataframe. All exceptions are caught and returned as an error message.
    
    Parameters:
        bars (pd.DataFrame): Input market bars; must be non-empty and contain a "close" column.
        symbol (str): Underlying symbol used for bar enrichment (uppercased internally).
        attach_vix (bool): If True, VIX data will be attached during enrichment.
        move_window (int): Forecast horizon window size (converted to int).
        vol_window (int): Volatility window size (converted to int).
        forecast_mode (ForecastMode): Forecast backend to run. One of:
            - "deterministic" — uses ForecastPipeline
            - "hmm" — builds a pipeline from `live` with model forced to "hmm"
            - "markov" — builds a pipeline from `live` with model forced to "markov"
            - "probabilistic" — uses ProbabilisticForecastPipeline (may use `probabilistic_model_path`)
        live (LiveRegimeModelConfig | None): Optional live regime configuration providing model/hyperparameters; defaults to a new LiveRegimeModelConfig() when omitted.
        probabilistic_model_path (str | Path | None): Optional path to a probabilistic model artifact; only used when `forecast_mode` is "probabilistic".
    
    Returns:
        tuple[pd.DataFrame | None, str | None]:
            On success, `(forecast_dataframe, None)` where `forecast_dataframe` is a copy of the pipeline output with `has_major_event = False`.
            On failure, `(None, error_message)` describing the validation error, unknown mode, or exception raised during processing.
    """
    if bars.empty or "close" not in bars.columns:
        return None, "No bars or missing 'close' column."

    cfg = live or LiveRegimeModelConfig()
    fcfg = cfg.forecast.to_forecast_config()
    mw, vw = int(move_window), int(vol_window)

    try:
        df = prepare_bars_for_factors(
            bars.copy(),
            option_chain=None,
            underlying=symbol.upper(),
            attach_vix=attach_vix,
        )
        feats = FactorPipeline().run(df)
        feats = classify_state_matrix(feats)

        if forecast_mode == "probabilistic":
            pipe = ProbabilisticForecastPipeline(
                config=fcfg,
                move_window=mw,
                vol_window=vw,
                model_path=probabilistic_model_path,
            )
            out = pipe.run(feats)
        elif forecast_mode == "deterministic":
            pipe = ForecastPipeline(config=fcfg, move_window=mw, vol_window=vw)
            out = pipe.run(feats)
        elif forecast_mode == "hmm":
            sub = cfg.model_copy(update={"model": "hmm"})
            built = sub.build_pipeline()
            assert isinstance(built, HybridForecastPipeline)
            out = built.run(feats)
        elif forecast_mode == "markov":
            sub = cfg.model_copy(update={"model": "markov"})
            built = sub.build_pipeline()
            assert isinstance(built, HybridMarkovForecastPipeline)
            out = built.run(feats)
        else:
            return None, f"Unknown forecast_mode: {forecast_mode!r}"

        out = out.copy()
        out["has_major_event"] = False
        return out, None
    except Exception as e:
        return None, str(e)
