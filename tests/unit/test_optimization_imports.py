from __future__ import annotations

import importlib


def test_optimization_tuning_compatibility_module_exports_helpers() -> None:
    feature_tuning = importlib.import_module("rlm.features.optimization.tuning")
    optimization_tuning = importlib.import_module("rlm.optimization.tuning")

    assert optimization_tuning.evaluate_forecast_backtest is feature_tuning.evaluate_forecast_backtest
    assert optimization_tuning.random_search_forecast_params is feature_tuning.random_search_forecast_params


def test_features_optimization_package_imports_without_missing_module() -> None:
    from rlm.features.optimization import evaluate_forecast_backtest, random_search_forecast_params

    assert callable(evaluate_forecast_backtest)
    assert callable(random_search_forecast_params)


def test_public_compatibility_import_paths_remain_available() -> None:
    from rlm.core.pipeline import FullRLMConfig as CanonicalFullRLMConfig
    from rlm.core.pipeline import FullRLMPipeline as CanonicalFullRLMPipeline
    from rlm.data.bars_enrichment import prepare_bars_for_factors as canonical_prepare_bars
    from rlm.data.microstructure.calculators.gex import gex_flip_level as canonical_gex_flip_level
    from rlm.data.microstructure.database.query import MicrostructureDB as CanonicalMicrostructureDB
    from rlm.data.microstructure.factors.gex_factors import GEXFactors as CanonicalGEXFactors
    from rlm.datasets.bars_enrichment import prepare_bars_for_factors
    from rlm.features.scoring.state_matrix import classify_state_matrix as canonical_classify_state
    from rlm.features.standardization.transforms import log_tanh_ratio as canonical_log_tanh_ratio
    from rlm.forecasting.kronos_config import KronosConfig as CanonicalKronosConfig
    from rlm.forecasting.kronos_forecast import KronosForecastPipeline as CanonicalKronosForecastPipeline
    from rlm.forecasting.models.kronos.model.kronos import Kronos as CanonicalKronos
    from rlm.forecasting.models.kronos.predictor import RLMKronosPredictor as CanonicalRLMKronosPredictor
    from rlm.forecasting.models.kronos.regime_confidence import (
        KronosRegimeConfidence as CanonicalKronosRegimeConfidence,
    )
    from rlm.kronos.config import KronosConfig
    from rlm.kronos.forecast import KronosForecastPipeline
    from rlm.kronos.model.kronos import Kronos
    from rlm.kronos.predictor import RLMKronosPredictor
    from rlm.kronos.regime_confidence import KronosRegimeConfidence
    from rlm.microstructure.calculators.gex import gex_flip_level
    from rlm.microstructure.database.query import MicrostructureDB
    from rlm.microstructure.factors.gex_factors import GEXFactors
    from rlm.pipeline import FullRLMConfig, FullRLMPipeline
    from rlm.scoring.state_matrix import classify_state_matrix
    from rlm.standardization.transforms import log_tanh_ratio

    assert FullRLMConfig is CanonicalFullRLMConfig
    assert FullRLMPipeline is CanonicalFullRLMPipeline
    assert prepare_bars_for_factors is canonical_prepare_bars
    assert KronosConfig is CanonicalKronosConfig
    assert KronosForecastPipeline is CanonicalKronosForecastPipeline
    assert KronosRegimeConfidence is CanonicalKronosRegimeConfidence
    assert RLMKronosPredictor is CanonicalRLMKronosPredictor
    assert Kronos is CanonicalKronos
    assert MicrostructureDB is CanonicalMicrostructureDB
    assert gex_flip_level is canonical_gex_flip_level
    assert GEXFactors is CanonicalGEXFactors
    assert classify_state_matrix is canonical_classify_state
    assert log_tanh_ratio is canonical_log_tanh_ratio
