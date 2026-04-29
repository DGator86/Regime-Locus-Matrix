from rlm.roee.decision import compute_regime_modulators


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
