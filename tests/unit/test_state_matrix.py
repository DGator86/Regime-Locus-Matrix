import pandas as pd

from rlm.features.scoring.state_matrix import classify_state_matrix


def test_state_matrix_creates_regime_columns() -> None:
    df = pd.DataFrame(
        {
            "S_D": [0.8, -0.8, 0.0, 0.4],
            "S_V": [-0.5, 0.7, -0.6, 0.0],
            "S_L": [0.6, -0.2, 0.8, 0.3],
            "S_G": [0.7, -0.8, 0.5, -0.1],
        }
    )

    out = classify_state_matrix(df)

    required = [
        "direction_regime",
        "volatility_regime",
        "liquidity_regime",
        "dealer_flow_regime",
        "regime_key",
    ]
    for col in required:
        assert col in out.columns

    assert out.loc[0, "direction_regime"] == "bull"
    assert out.loc[1, "direction_regime"] == "bear"
    assert out.loc[2, "direction_regime"] == "range"
