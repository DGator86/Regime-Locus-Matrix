from __future__ import annotations

import importlib


def test_optimization_tuning_compatibility_module_exports_helpers() -> None:
    feature_tuning = importlib.import_module("rlm.features.optimization.tuning")
    optimization_tuning = importlib.import_module("rlm.optimization.tuning")

    assert (
        optimization_tuning.evaluate_forecast_backtest is feature_tuning.evaluate_forecast_backtest
    )
    assert (
        optimization_tuning.random_search_forecast_params
        is feature_tuning.random_search_forecast_params
    )


def test_features_optimization_package_imports_without_missing_module() -> None:
    from rlm.features.optimization import evaluate_forecast_backtest, random_search_forecast_params

    assert callable(evaluate_forecast_backtest)
    assert callable(random_search_forecast_params)
