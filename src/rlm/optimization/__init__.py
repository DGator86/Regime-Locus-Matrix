"""Backward-compatibility re-export. Canonical location: rlm.features.optimization.  (PR #41)"""

from rlm.features.optimization.tuning import evaluate_forecast_backtest, random_search_forecast_params

__all__ = ["evaluate_forecast_backtest", "random_search_forecast_params"]
