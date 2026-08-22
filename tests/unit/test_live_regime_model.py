from __future__ import annotations

import pandas as pd
import pytest

from rlm.features.scoring.state_matrix import classify_state_matrix
from rlm.forecasting.live_model import (
    LiveForecastParameters,
    LiveKronosParameters,
    LiveRegimeModelConfig,
    LiveROEEParameters,
    LiveTimeframeHierarchy,
    load_live_regime_model,
    preserve_operational_live_fields,
    promote_live_regime_model,
    save_live_regime_model,
)
from rlm.roee.decision import select_trade_for_row


def _synthetic_scores(n: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    close = pd.Series(range(n), index=idx, dtype=float).cumsum() + 5000.0
    df = pd.DataFrame(
        {
            "close": close,
            "sigma": 0.02,
            "S_D": (pd.Series(range(n), index=idx) % 11).astype(float) / 10.0,
            "S_V": (pd.Series(range(n), index=idx) % 7).astype(float) / 10.0,
            "S_L": (pd.Series(range(n), index=idx) % 13).astype(float) / 10.0,
            "S_G": (pd.Series(range(n), index=idx) % 5).astype(float) / 10.0,
        },
        index=idx,
    )
    return classify_state_matrix(df)


def test_select_trade_for_row_uses_markov_probabilities() -> None:
    row = pd.Series(
        {
            "close": 505.0,
            "sigma": 0.02,
            "S_D": 0.6,
            "S_V": -0.4,
            "S_L": 0.2,
            "S_G": 0.3,
            "direction_regime": "bullish",
            "volatility_regime": "normal",
            "liquidity_regime": "liquid",
            "dealer_flow_regime": "supportive",
            "regime_key": "bullish_normal_liquid_supportive",
            "markov_probs": [0.45, 0.35, 0.20],
        }
    )

    decision = select_trade_for_row(
        row,
        strike_increment=5.0,
        hmm_confidence_threshold=0.6,
    )

    assert decision.action == "skip"
    assert decision.strategy_name == "markov_gate"
    assert decision.metadata["regime_model"] == "markov"
    assert decision.metadata["regime_trade_allowed"] is False


def test_live_regime_model_timeframe_hierarchy_defaults_roundtrip(tmp_path) -> None:
    cfg = LiveRegimeModelConfig(model="forecast")
    assert cfg.timeframe_hierarchy.confirmation_bar_sizes == ()
    path = tmp_path / "live_regime_model.json"
    save_live_regime_model(cfg, path)
    loaded = load_live_regime_model(path)
    assert loaded.timeframe_hierarchy.confirmation_duration == "10 D"


def test_live_regime_model_timeframe_hierarchy_with_confirmations_roundtrip(tmp_path) -> None:
    cfg = LiveRegimeModelConfig(
        model="forecast",
        timeframe_hierarchy=LiveTimeframeHierarchy(
            primary_bar_size="1 day",
            primary_duration="252 D",
            confirmation_bar_sizes=("15 mins", "5 mins"),
            confirmation_mode="both",
            require_all_confirmations=False,
        ),
    )
    path = tmp_path / "live_regime_model.json"
    save_live_regime_model(cfg, path)
    loaded = load_live_regime_model(path)
    assert loaded.timeframe_hierarchy.primary_bar_size == "1 day"
    assert loaded.timeframe_hierarchy.confirmation_bar_sizes == ("15 mins", "5 mins")
    assert loaded.timeframe_hierarchy.require_all_confirmations is False


@pytest.mark.filterwarnings(
    "ignore:Invalid regime transition probabilities estimated in EM iteration:statsmodels.tools.sm_exceptions.EstimationWarning"
)
@pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_live_regime_model_round_trip_builds_markov_pipeline(tmp_path) -> None:
    cfg = LiveRegimeModelConfig(
        model="markov",
        provenance={"source": "unit-test"},
    )
    path = tmp_path / "live_regime_model.json"
    save_live_regime_model(cfg, path)
    loaded = load_live_regime_model(path)

    df = _synthetic_scores()
    train_mask = pd.Series(df.index < df.index[180], index=df.index)
    out = loaded.build_pipeline().run(df, train_mask=train_mask)

    assert loaded.model == "markov"
    assert loaded.provenance["source"] == "unit-test"
    assert "markov_probs" in out.columns
    assert "markov_state" in out.columns


def _vps_tuned_live_model() -> LiveRegimeModelConfig:
    return LiveRegimeModelConfig(
        model="markov",
        forecast=LiveForecastParameters(move_window=61, vol_window=141),
        roee=LiveROEEParameters(confidence_threshold=0.4, kronos_transition_penalty=0.25),
        use_kronos=True,
        kronos=LiveKronosParameters(model_name="NeoQuasar/Kronos-mini", weight=0.35),
        timeframe_hierarchy=LiveTimeframeHierarchy(
            primary_bar_size="1 min",
            primary_duration="30 D",
            confirmation_bar_sizes=("5 mins", "15 mins"),
        ),
        min_regime_train_samples=12,
    )


def test_fresh_promote_config_drops_operational_overlays() -> None:
    """Document the champion-only object that weekly calibrate used to save raw."""
    promoted = LiveRegimeModelConfig(
        model="hmm",
        forecast=LiveForecastParameters(move_window=80, vol_window=90),
        provenance={"selection_metric": "sharpe"},
    )
    assert promoted.use_kronos is False
    assert promoted.timeframe_hierarchy.confirmation_bar_sizes == ()
    assert promoted.roee.confidence_threshold == 0.6
    assert promoted.min_regime_train_samples is None


def test_preserve_operational_live_fields_keeps_kronos_mtf_and_roee() -> None:
    existing = _vps_tuned_live_model()
    promoted = LiveRegimeModelConfig(
        model="hmm",
        forecast=LiveForecastParameters(move_window=80, vol_window=90),
        provenance={"selection_metric": "sharpe"},
    )
    merged = preserve_operational_live_fields(promoted, existing)

    assert merged.model == "hmm"
    assert merged.forecast.move_window == 80
    assert merged.forecast.vol_window == 90
    assert merged.use_kronos is True
    assert merged.kronos.model_name == "NeoQuasar/Kronos-mini"
    assert merged.timeframe_hierarchy.confirmation_bar_sizes == ("5 mins", "15 mins")
    assert merged.timeframe_hierarchy.primary_bar_size == "1 min"
    assert merged.roee.confidence_threshold == 0.4
    assert merged.roee.kronos_transition_penalty == 0.25
    assert merged.min_regime_train_samples == 12


def test_promote_live_regime_model_preserves_existing_overlays(tmp_path) -> None:
    path = tmp_path / "live_regime_model.json"
    save_live_regime_model(_vps_tuned_live_model(), path)

    promoted = LiveRegimeModelConfig(
        model="hmm",
        forecast=LiveForecastParameters(
            drift_gamma_alpha=0.7,
            sigma_floor=2e-4,
            direction_neutral_threshold=0.25,
            move_window=80,
            vol_window=90,
        ),
        provenance={"selection_metric": "sharpe", "trials": 24},
    )
    written = promote_live_regime_model(promoted, path)
    loaded = load_live_regime_model(path)

    assert written.model == "hmm"
    assert loaded.model == "hmm"
    assert loaded.forecast.move_window == 80
    assert loaded.use_kronos is True
    assert loaded.kronos.weight == 0.35
    assert loaded.timeframe_hierarchy.confirmation_bar_sizes == ("5 mins", "15 mins")
    assert loaded.roee.confidence_threshold == 0.4
    assert loaded.min_regime_train_samples == 12


def test_promote_live_regime_model_without_existing_file_keeps_champion(tmp_path) -> None:
    path = tmp_path / "live_regime_model.json"
    promoted = LiveRegimeModelConfig(model="hmm", provenance={"source": "first-promote"})
    written = promote_live_regime_model(promoted, path)
    loaded = load_live_regime_model(path)

    assert written.use_kronos is False
    assert loaded.use_kronos is False
    assert loaded.model == "hmm"
    assert loaded.provenance["source"] == "first-promote"
