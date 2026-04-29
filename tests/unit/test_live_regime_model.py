from __future__ import annotations

import pandas as pd

from rlm.forecasting.live_model import (
    LiveRegimeModelConfig,
    load_live_regime_model,
    save_live_regime_model,
)
from rlm.roee.decision import select_trade_for_row
from rlm.scoring.state_matrix import classify_state_matrix


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
