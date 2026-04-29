import pandas as pd

from rlm.roee.decision import compute_regime_modulators, select_trade_for_row


def test_epistemic_uncertainty_disables_trade_when_threshold_hit() -> None:
    row = {
        "hmm_probs": [0.9, 0.1],
        "kronos_regime_agreement": 0.8,
        "kronos_epistemic_uncertainty": 0.95,
    }
    out = compute_regime_modulators(
        row,
        confidence_threshold=0.6,
        sizing_multiplier=1.0,
        transition_penalty=0.5,
        kronos_epistemic_disable_threshold=0.7,
    )
    assert out["trade"] is False


def test_aleatoric_uncertainty_reduces_size_multiplier() -> None:
    base_row = {"hmm_probs": [0.9, 0.1], "kronos_regime_agreement": 0.8}
    noisy_row = {
        "hmm_probs": [0.9, 0.1],
        "kronos_regime_agreement": 0.8,
        "kronos_aleatoric_uncertainty": 0.9,
    }
    base = compute_regime_modulators(
        base_row,
        confidence_threshold=0.6,
        sizing_multiplier=1.0,
        transition_penalty=0.5,
        kronos_aleatoric_size_penalty=0.5,
    )
    noisy = compute_regime_modulators(
        noisy_row,
        confidence_threshold=0.6,
        sizing_multiplier=1.0,
        transition_penalty=0.5,
        kronos_aleatoric_size_penalty=0.5,
    )
    assert noisy["size_mult"] < base["size_mult"]


def test_select_trade_for_row_applies_kronos_epistemic_gate() -> None:
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
            "hmm_probs": [0.9, 0.1],
            "kronos_regime_agreement": 0.9,
            "kronos_epistemic_uncertainty": 0.95,
        }
    )

    decision = select_trade_for_row(
        row,
        strike_increment=5.0,
        hmm_confidence_threshold=0.6,
        kronos_epistemic_disable_threshold=0.7,
    )

    assert decision.action == "skip"
    assert decision.metadata["regime_trade_allowed"] is False
    assert decision.metadata["regime_model"] == "hmm+kronos+epi_gate"
