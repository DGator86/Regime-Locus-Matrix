"""Smoke test: end-to-end FullRLMPipeline run on synthetic fixture data.

This is the single highest-value test for launch confidence.  It exercises:
  bars → FactorPipeline → ForecastPipeline (HMM) → ROEE policy

No network calls, no external files — fully deterministic via the synthetic
bar generator that's already used throughout the unit test suite.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.datasets.backtest_data import synthetic_bars_demo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bars_df() -> pd.DataFrame:
    return synthetic_bars_demo(end=pd.Timestamp("2024-12-31"), periods=220)


@pytest.fixture(scope="module")
def pipeline_result(bars_df: pd.DataFrame) -> PipelineResult:
    cfg = FullRLMConfig(
        symbol="TEST",
        regime_model="hmm",
        hmm_states=4,
        use_kronos=False,  # avoid network / model weight download in CI
        attach_vix=False,  # no yfinance call
        run_backtest=False,
    )
    return FullRLMPipeline(cfg).run(bars_df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factors_df_not_empty(pipeline_result: PipelineResult) -> None:
    assert not pipeline_result.factors_df.empty, "factors_df should not be empty"


def test_forecast_df_not_empty(pipeline_result: PipelineResult) -> None:
    assert not pipeline_result.forecast_df.empty, "forecast_df should not be empty"


def test_policy_df_not_empty(pipeline_result: PipelineResult) -> None:
    assert not pipeline_result.policy_df.empty, "policy_df should not be empty"


def test_factors_df_has_score_columns(pipeline_result: PipelineResult) -> None:
    score_cols = {"S_D", "S_V", "S_L", "S_G"}
    missing = score_cols - set(pipeline_result.factors_df.columns)
    assert not missing, f"factors_df missing score columns: {missing}"


def test_forecast_df_has_regime_columns(pipeline_result: PipelineResult) -> None:
    regime_cols = {"hmm_state", "hmm_state_label"}
    missing = regime_cols - set(pipeline_result.forecast_df.columns)
    assert not missing, f"forecast_df missing regime columns: {missing}"


def test_policy_df_has_action_columns(pipeline_result: PipelineResult) -> None:
    action_cols = {"roee_action", "roee_strategy", "roee_size_fraction"}
    missing = action_cols - set(pipeline_result.policy_df.columns)
    assert not missing, f"policy_df missing action columns: {missing}"


def test_policy_df_last_row_has_action(pipeline_result: PipelineResult) -> None:
    last = pipeline_result.policy_df.iloc[-1]
    assert pd.notna(last["roee_action"]), "Last policy row roee_action should not be null"


def test_no_backtest_fields_without_flag(pipeline_result: PipelineResult) -> None:
    assert pipeline_result.backtest_trades is None
    assert pipeline_result.backtest_equity is None
    assert pipeline_result.backtest_metrics is None


def test_factors_df_length_matches_bars(bars_df: pd.DataFrame, pipeline_result: PipelineResult) -> None:
    assert len(pipeline_result.factors_df) <= len(bars_df), "factors_df should not be longer than input bars"
    assert len(pipeline_result.factors_df) > 0


def test_policy_df_length_matches_forecast(pipeline_result: PipelineResult) -> None:
    assert len(pipeline_result.policy_df) == len(
        pipeline_result.forecast_df
    ), "policy_df and forecast_df should have the same row count"
