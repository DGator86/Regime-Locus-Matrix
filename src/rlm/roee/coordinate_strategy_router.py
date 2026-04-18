from __future__ import annotations

from typing import Mapping

from rlm.roee.strategy_value_model import StrategyValueModel

_DEFAULT_VALUE_MODEL = StrategyValueModel.with_bootstrap_coefficients()


def select_strategy_from_coordinates(row: Mapping[str, float]) -> str:
    scores = _DEFAULT_VALUE_MODEL.score_row(row)
    return scores.best_strategy
