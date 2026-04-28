from __future__ import annotations


def test_optimization_tuning_compatibility_module_exports_helpers() -> None:
    from rlm.features.optimization.tuning import (
        evaluate_forecast_backtest as feature_evaluate_forecast_backtest,
        random_search_forecast_params as feature_random_search_forecast_params,
    )
    from rlm.optimization.tuning import evaluate_forecast_backtest, random_search_forecast_params

    assert evaluate_forecast_backtest is feature_evaluate_forecast_backtest
    assert random_search_forecast_params is feature_random_search_forecast_params


def test_features_optimization_package_imports_without_missing_module() -> None:
    from rlm.features.optimization import evaluate_forecast_backtest, random_search_forecast_params

    assert callable(evaluate_forecast_backtest)
    assert callable(random_search_forecast_params)
