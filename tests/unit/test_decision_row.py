import pandas as pd

from rlm.roee.decision import select_trade_for_row


def test_select_trade_for_row_skips_when_required_columns_missing() -> None:
    row = pd.Series({"close": 100.0, "sigma": 0.01})
    d = select_trade_for_row(row, strike_increment=1.0)
    assert d.action == "skip"
    assert "Missing required row columns" in (d.rationale or "")
    assert d.metadata.get("missing_columns")


def test_select_trade_for_row_runs_when_columns_present() -> None:
    row = pd.Series(
        {
            "close": 5000.0,
            "sigma": 0.01,
            "S_D": 0.8,
            "S_V": -0.5,
            "S_L": 0.7,
            "S_G": 0.8,
            "direction_regime": "bull",
            "volatility_regime": "low_vol",
            "liquidity_regime": "high_liquidity",
            "dealer_flow_regime": "supportive",
            "regime_key": "bull|low_vol|high_liquidity|supportive",
        }
    )
    d = select_trade_for_row(row, strike_increment=5.0)
    assert d.action == "enter"
    assert d.size_fraction is not None
    assert d.size_fraction == round(float(d.size_fraction), 10)


def test_select_trade_for_row_can_use_dynamic_sizing() -> None:
    row = pd.Series(
        {
            "close": 5000.0,
            "sigma": 0.01,
            "S_D": 0.8,
            "S_V": -0.5,
            "S_L": 0.7,
            "S_G": 0.8,
            "direction_regime": "bull",
            "volatility_regime": "low_vol",
            "liquidity_regime": "high_liquidity",
            "dealer_flow_regime": "supportive",
            "regime_key": "bull|low_vol|high_liquidity|supportive",
            "forecast_return": 0.02,
            "realized_vol": 0.20,
        }
    )
    d = select_trade_for_row(
        row,
        strike_increment=5.0,
        use_dynamic_sizing=True,
    )
    assert d.action == "enter"
    assert d.size_fraction == 0.03
    assert d.metadata.get("size_model") == "kelly_vol_target"


def test_select_trade_for_row_can_apply_vault_to_dynamic_sizing() -> None:
    row = pd.Series(
        {
            "close": 5000.0,
            "sigma": 0.01,
            "S_D": 0.8,
            "S_V": -0.5,
            "S_L": 0.7,
            "S_G": 0.8,
            "direction_regime": "bull",
            "volatility_regime": "low_vol",
            "liquidity_regime": "high_liquidity",
            "dealer_flow_regime": "supportive",
            "regime_key": "bull|low_vol|high_liquidity|supportive",
            "forecast_return": 0.02,
            "forecast_uncertainty": 0.04,
            "realized_vol": 0.20,
        }
    )
    d = select_trade_for_row(
        row,
        strike_increment=5.0,
        use_dynamic_sizing=True,
        vault_uncertainty_threshold=0.03,
        vault_size_multiplier=0.5,
    )
    assert d.action == "enter"
    assert d.size_fraction == 0.015
    assert d.metadata.get("size_model") == "kelly_vol_target"
    assert d.metadata.get("vault_triggered") is True


def test_select_trade_for_row_uses_markov_state_for_regime_adjusted_kelly() -> None:
    row = pd.Series(
        {
            "close": 5000.0,
            "sigma": 0.01,
            "S_D": 0.8,
            "S_V": -0.5,
            "S_L": 0.7,
            "S_G": 0.8,
            "direction_regime": "bull",
            "volatility_regime": "low_vol",
            "liquidity_regime": "high_liquidity",
            "dealer_flow_regime": "supportive",
            "regime_key": "bull|low_vol|high_liquidity|supportive",
            "forecast_return": 0.5,
            "realized_vol": 1.5,
            "markov_state_label": "bear|high_vol|high_liquidity|supportive_like",
            "markov_confidence": 0.8,
            "hmm_state_label": "bull|low_vol|high_liquidity|supportive_like",
            "hmm_probs": [0.9, 0.1],
        }
    )
    d = select_trade_for_row(
        row,
        strike_increment=5.0,
        use_dynamic_sizing=True,
    )

    assert d.action == "enter"
    assert d.size_fraction == 0.0108
    assert d.metadata.get("kelly_latent_regime_source") == "markov"
    assert d.metadata.get("regime_state_label") == "bear|high_vol|high_liquidity|supportive_like"
