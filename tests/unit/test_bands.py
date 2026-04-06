import pandas as pd

from rlm.forecasting.bands import compute_state_matrix_bands


def test_bands_floor_half_width_for_tiny_sigma() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0],
            "mean_price": [100.0],
            "sigma": [1e-12],
        }
    )
    out = compute_state_matrix_bands(df)
    w = float(out["core_zone_width"].iloc[0])
    # sigma negligible → half-width floors to max(close*1e-5, 1e-8) = 0.001 → width 0.002
    assert w >= 0.0019
    assert "band_half_width_1s" in out.columns


def test_bands_normal_sigma_unchanged_magnitude_order() -> None:
    df = pd.DataFrame(
        {
            "close": [400.0],
            "mean_price": [400.0],
            "sigma": [0.02],
        }
    )
    out = compute_state_matrix_bands(df)
    half = float(out["band_half_width_1s"].iloc[0])
    assert abs(half - 8.0) < 1e-9
