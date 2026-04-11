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
    Mirror ``scripts/rlm_terminal.run_rlm_pipeline`` with extra forecast backends.

    Parameters
    ----------
    forecast_mode
        * ``deterministic`` — :class:`ForecastPipeline`
        * ``hmm`` / ``markov`` — :meth:`LiveRegimeModelConfig.build_pipeline` with
          ``model`` forced accordingly
        * ``probabilistic`` — :class:`ProbabilisticForecastPipeline` (optional JSON artifact)
    live
        Used for HMM/Markov/forecast hyperparameters; defaults to a fresh
        :class:`LiveRegimeModelConfig` when omitted.
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
