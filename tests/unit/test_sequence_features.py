from __future__ import annotations

import pandas as pd

from rlm.training.sequence_features import add_sequence_features


def test_sequence_features_add_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "M_D": [5.0, 6.0, 7.0],
            "M_V": [4.0, 4.5, 5.0],
            "M_L": [6.0, 6.0, 6.0],
            "M_G": [5.0, 5.5, 6.0],
            "M_trend_strength": [0.0, 1.0, 2.0],
            "M_dealer_control": [0.0, 0.5, 1.0],
            "M_alignment": [0.0, 0.5, 2.0],
            "M_delta_neutral": [1.0, 2.0, 3.0],
            "M_R_trans": [0.0, 1.0, 1.0],
        }
    )
    out = add_sequence_features(df, window=3)
    assert "M_D_mean_3" in out.columns
    assert "M_alignment_std_3" in out.columns
    assert "M_D_slope_3" in out.columns
