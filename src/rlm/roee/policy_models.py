from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rlm.features.scoring.regime_model import RegimeModel
from rlm.roee.strategy_value_model import StrategyValueModel


@dataclass(frozen=True)
class TradeDecision:
    trade: bool
    strategy: str
    size_fraction: float
    regime_probabilities: dict[str, float]
    expected_values: dict[str, float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PolicyConstraints:
    trade_probability_threshold: float = 0.55
    min_edge: float = 0.0
    max_size_fraction: float = 0.25
    kappa: float = 0.15
    epsilon: float = 1e-6


def select_trade_from_models(
    row: Mapping[str, float],
    regime_model: RegimeModel,
    value_model: StrategyValueModel,
    constraints: PolicyConstraints,
) -> TradeDecision:
    regime_probs_arr = regime_model.predict_proba(
        np.array(
            [
                [
                    float(row.get("M_D", 0.0)),
                    float(row.get("M_V", 0.0)),
                    float(row.get("M_L", 0.0)),
                    float(row.get("M_G", 0.0)),
                    float(row.get("M_trend_strength", 0.0)),
                    float(row.get("M_dealer_control", 0.0)),
                    float(row.get("M_alignment", 0.0)),
                    float(row.get("M_delta_neutral", 0.0)),
                    float(row.get("M_R_trans", 0.0)),
                ]
            ],
            dtype=float,
        )
    )[0]
    regime_probs = {regime_model.labels[i]: float(regime_probs_arr[i]) for i in range(len(regime_model.labels))}

    strategy_scores = value_model.score_row(row)
    best_strategy = strategy_scores.best_strategy
    best_edge = strategy_scores.scores[best_strategy]

    trade_probability = 1.0 - regime_probs.get("no_trade", 0.0)
    transition_penalty = 1.0 - regime_probs.get("transition", 0.0)
    uncertainty = max(float(np.std(list(strategy_scores.scores.values()))), constraints.epsilon)

    trade_allowed = trade_probability >= constraints.trade_probability_threshold and best_edge >= constraints.min_edge
    raw_size = constraints.kappa * (best_edge / uncertainty) * transition_penalty
    size_fraction = min(max(raw_size, 0.0), constraints.max_size_fraction) if trade_allowed else 0.0

    chosen_strategy = best_strategy if trade_allowed else "no_trade"
    return TradeDecision(
        trade=trade_allowed,
        strategy=chosen_strategy,
        size_fraction=float(size_fraction),
        regime_probabilities=regime_probs,
        expected_values=strategy_scores.scores,
        metadata={
            "trade_probability": trade_probability,
            "transition_penalty": transition_penalty,
            "best_edge": best_edge,
            "uncertainty": uncertainty,
        },
    )
