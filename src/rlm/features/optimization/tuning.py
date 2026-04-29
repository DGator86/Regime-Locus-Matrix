"""Random search over forecast + rolling-window parameters with one factor pass."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from rlm.backtest.engine import BacktestEngine
from rlm.features.scoring.state_matrix import classify_state_matrix
from rlm.forecasting.engines import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridMarkovForecastPipeline,
)
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.roee.engine import ROEEConfig
from rlm.types.forecast import ForecastConfig

ObjectiveName = Literal["sharpe", "calmar", "composite", "total_return"]
RegimeModelName = Literal["forecast", "hmm", "markov"]


def objective_value(
    summary: dict[str, float],
    name: ObjectiveName,
    *,
    min_trades: int,
    drawdown_penalty: float = 0.75,
) -> float:
    """Higher is better. Returns -inf if constraints fail."""
    n = int(summary.get("num_trades", 0) or 0)
    if n < min_trades:
        return float("-inf")

    if name == "total_return":
        tr = float(summary.get("total_return_pct", float("nan")))
        return tr if math.isfinite(tr) else float("-inf")

    if name == "sharpe":
        sh = float(summary.get("sharpe", float("nan")))
        return sh if math.isfinite(sh) else float("-inf")

    if name == "calmar":
        tr = float(summary.get("total_return_pct", float("nan")))
        dd = float(summary.get("max_drawdown", float("nan")))
        if not math.isfinite(tr) or not math.isfinite(dd) or dd >= -1e-9:
            return float("-inf")
        return tr / abs(dd)

    # composite: Sharpe + penalty * max_drawdown (drawdown is negative)
    sh = float(summary.get("sharpe", float("nan")))
    dd = float(summary.get("max_drawdown", float("nan")))
    if not math.isfinite(sh):
        sh = -2.0
    if not math.isfinite(dd):
        dd = -1.0
    return sh + drawdown_penalty * dd


@dataclass(frozen=True)
class ForecastParamSample:
    drift_gamma_alpha: float
    sigma_floor: float
    direction_neutral_threshold: float
    move_window: int
    vol_window: int


def generate_forecast_param_samples(
    *,
    n_trials: int,
    rng: np.random.Generator,
) -> list[ForecastParamSample]:
    return [
        ForecastParamSample(
            drift_gamma_alpha=float(rng.uniform(0.35, 0.95)),
            sigma_floor=float(10 ** rng.uniform(-5.0, -2.5)),
            direction_neutral_threshold=float(rng.uniform(0.12, 0.48)),
            move_window=int(rng.integers(50, 181)),
            vol_window=int(rng.integers(50, 181)),
        )
        for _ in range(n_trials)
    ]


def _resolve_regime_model(
    *,
    regime_model: RegimeModelName | None,
    use_hmm: bool,
    use_markov: bool,
) -> RegimeModelName:
    if regime_model is not None:
        return regime_model
    if use_hmm and use_markov:
        raise ValueError("Choose only one regime model for tuning.")
    if use_hmm:
        return "hmm"
    if use_markov:
        return "markov"
    return "forecast"


def evaluate_forecast_backtest(
    factor_frame: pd.DataFrame,
    option_chain: pd.DataFrame,
    *,
    underlying_symbol: str,
    sample: ForecastParamSample,
    regime_model: RegimeModelName | None = None,
    use_hmm: bool = False,
    use_markov: bool = False,
    hmm_states: int = 6,
    hmm_n_iter: int = 100,
    hmm_filter_backend: Literal["auto", "numpy", "numba"] = "auto",
    hmm_prefer_gpu: bool = False,
    markov_states: int = 3,
    roee_config: ROEEConfig | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Run forecast → state matrix → :class:`~rlm.backtest.engine.BacktestEngine` for one parameter set.

    ``factor_frame`` must be the output of :class:`~rlm.factors.pipeline.FactorPipeline` (scores + raw).
    """
    fc = ForecastConfig(
        drift_gamma_alpha=sample.drift_gamma_alpha,
        sigma_floor=sample.sigma_floor,
        direction_neutral_threshold=sample.direction_neutral_threshold,
    )
    selected_model = _resolve_regime_model(
        regime_model=regime_model,
        use_hmm=use_hmm,
        use_markov=use_markov,
    )
    if selected_model == "hmm":
        pipe = HybridForecastPipeline(
            config=fc,
            move_window=sample.move_window,
            vol_window=sample.vol_window,
            hmm_config=HMMConfig(
                n_states=hmm_states,
                n_iter=hmm_n_iter,
                filter_backend=hmm_filter_backend,
                prefer_gpu=hmm_prefer_gpu,
            ),
        )
        features = pipe.run(factor_frame.copy())
    elif selected_model == "markov":
        pipe = HybridMarkovForecastPipeline(
            config=fc,
            move_window=sample.move_window,
            vol_window=sample.vol_window,
            markov_config=MarkovSwitchingConfig(n_states=markov_states),
        )
        features = pipe.run(factor_frame.copy())
    else:
        pipe = ForecastPipeline(
            config=fc,
            move_window=sample.move_window,
            vol_window=sample.vol_window,
        )
        features = pipe.run(factor_frame.copy())

    features = classify_state_matrix(features)

    engine = BacktestEngine(
        initial_capital=100_000.0,
        contract_multiplier=100,
        strike_increment=5.0,
        underlying_symbol=underlying_symbol,
        quantity_per_trade=1,
        roee_config=roee_config or (ROEEConfig() if selected_model in {"hmm", "markov"} else None),
    )
    _, trades, summary = engine.run(features, option_chain)

    params: dict[str, Any] = {
        "drift_gamma_alpha": sample.drift_gamma_alpha,
        "sigma_floor": sample.sigma_floor,
        "direction_neutral_threshold": sample.direction_neutral_threshold,
        "move_window": sample.move_window,
        "vol_window": sample.vol_window,
        "regime_model": selected_model,
        "use_hmm": selected_model == "hmm",
        "use_markov": selected_model == "markov",
    }
    if selected_model == "hmm":
        params["hmm_states"] = hmm_states
        params["hmm_n_iter"] = hmm_n_iter
        params["hmm_filter_backend"] = hmm_filter_backend
        params["hmm_prefer_gpu"] = hmm_prefer_gpu
    if selected_model == "markov":
        params["markov_states"] = markov_states
    params["num_trades"] = int(len(trades))
    return summary, params


def random_search_forecast_params(
    factor_frame: pd.DataFrame,
    option_chain: pd.DataFrame,
    *,
    underlying_symbol: str,
    n_trials: int,
    rng: np.random.Generator,
    objective: ObjectiveName = "composite",
    min_trades: int = 15,
    regime_model: RegimeModelName | None = None,
    use_hmm: bool = False,
    use_markov: bool = False,
    hmm_states: int = 6,
    hmm_n_iter: int = 100,
    hmm_filter_backend: Literal["auto", "numpy", "numba"] = "auto",
    hmm_prefer_gpu: bool = False,
    markov_states: int = 3,
    drawdown_penalty: float = 0.75,
    samples: list[ForecastParamSample] | None = None,
    roee_config: ROEEConfig | None = None,
) -> list[tuple[float, dict[str, float], dict[str, Any]]]:
    """
    Random search. Returns list of ``(objective_score, summary, params)`` sorted best-first.
    """
    rows: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    selected_model = _resolve_regime_model(
        regime_model=regime_model,
        use_hmm=use_hmm,
        use_markov=use_markov,
    )
    param_samples = samples or generate_forecast_param_samples(
        n_trials=max(1, n_trials),
        rng=rng,
    )

    for sample in param_samples:
        summary, params = evaluate_forecast_backtest(
            factor_frame,
            option_chain,
            underlying_symbol=underlying_symbol,
            sample=sample,
            regime_model=selected_model,
            use_hmm=selected_model == "hmm",
            use_markov=selected_model == "markov",
            hmm_states=hmm_states,
            hmm_n_iter=hmm_n_iter,
            hmm_filter_backend=hmm_filter_backend,
            hmm_prefer_gpu=hmm_prefer_gpu,
            markov_states=markov_states,
            roee_config=roee_config,
        )
        score = objective_value(
            summary,
            objective,
            min_trades=min_trades,
            drawdown_penalty=drawdown_penalty,
        )
        rows.append((score, summary, params))

    rows.sort(key=lambda x: x[0], reverse=True)
    return rows
