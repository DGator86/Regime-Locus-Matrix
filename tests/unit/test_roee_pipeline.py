import numpy as np
import pandas as pd

from rlm.roee.pipeline import ROEEConfig, apply_roee_policy


def test_apply_roee_policy_surfaces_vault_scaling_columns() -> None:
    frame = pd.DataFrame(
        [
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
                "forecast_uncertainty": 0.04,
                "hmm_probs": np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.02]),
            }
        ],
        index=pd.to_datetime(["2024-01-02"]),
    )

    vaulted = apply_roee_policy(
        frame,
        strike_increment=5.0,
        config=ROEEConfig(vault_uncertainty_threshold=0.03, vault_size_multiplier=0.5),
    )
    baseline = apply_roee_policy(
        frame,
        strike_increment=5.0,
        config=ROEEConfig(vault_uncertainty_threshold=None, vault_size_multiplier=0.5),
    )

    assert vaulted.loc[frame.index[0], "roee_action"] == "enter"
    assert vaulted.loc[frame.index[0], "vault_triggered"]
    assert vaulted.loc[frame.index[0], "vault_size_multiplier"] == 0.5
    assert vaulted.loc[frame.index[0], "vault_forecast_uncertainty"] == 0.04
    assert vaulted.loc[frame.index[0], "vault_uncertainty_threshold"] == 0.03
    assert vaulted.loc[frame.index[0], "roee_size_fraction"] == baseline.loc[frame.index[0], "roee_size_fraction"] * 0.5


def test_apply_roee_policy_daily_circuit_breaker_blocks_trades() -> None:
    frame = pd.DataFrame(
        [
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
                "hmm_probs": np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.02]),
                "pnl_pct": -0.03,
            },
            {
                "close": 5010.0,
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
                "hmm_probs": np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.02]),
                "pnl_pct": -0.01,
            },
        ],
        index=pd.to_datetime(["2024-01-02 00:00:00", "2024-01-02 01:00:00"]),
    )
    out = apply_roee_policy(frame, config=ROEEConfig(daily_loss_circuit_breaker_pct=-0.02))
    assert out.iloc[0]["roee_action"] == "hold"
    assert out.iloc[0]["roee_strategy"] == "circuit_breaker"


def test_apply_roee_policy_correlation_haircut_reduces_size() -> None:
    frame = pd.DataFrame(
        [
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
                "hmm_probs": np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.02]),
                "portfolio_corr_exposure": 0.95,
            }
        ],
        index=pd.to_datetime(["2024-01-02"]),
    )
    base = apply_roee_policy(frame, config=ROEEConfig(correlation_exposure_threshold=None))
    cut = apply_roee_policy(
        frame,
        config=ROEEConfig(correlation_exposure_threshold=0.8, correlation_exposure_haircut=0.5),
    )
    assert cut.iloc[0]["roee_size_fraction"] == base.iloc[0]["roee_size_fraction"] * 0.5
