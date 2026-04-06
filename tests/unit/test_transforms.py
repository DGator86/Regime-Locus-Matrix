import math

import numpy as np
import pandas as pd

from rlm.standardization.transforms import log_tanh_ratio, log_tanh_signed, sigma_floor


def test_log_tanh_ratio_neutral_is_zero() -> None:
    assert math.isclose(log_tanh_ratio(1.0, 1.0, k=1.0), 0.0, abs_tol=1e-12)


def test_log_tanh_ratio_above_neutral_positive() -> None:
    assert log_tanh_ratio(2.0, 1.0, k=1.0) > 0.0


def test_log_tanh_ratio_below_neutral_negative() -> None:
    assert log_tanh_ratio(0.5, 1.0, k=1.0) < 0.0


def test_log_tanh_ratio_invert_flips_sign() -> None:
    normal = log_tanh_ratio(2.0, 1.0, k=1.0, invert=False)
    inverted = log_tanh_ratio(2.0, 1.0, k=1.0, invert=True)
    assert math.isclose(inverted, -normal, rel_tol=1e-12)


def test_log_tanh_signed_neutral_is_zero() -> None:
    assert math.isclose(log_tanh_signed(0.0, 1.0, k=1.0), 0.0, abs_tol=1e-12)


def test_log_tanh_signed_positive_is_positive() -> None:
    assert log_tanh_signed(1.0, 1.0, k=1.0) > 0.0


def test_log_tanh_signed_negative_is_negative() -> None:
    assert log_tanh_signed(-1.0, 1.0, k=1.0) < 0.0


def test_transform_outputs_bounded() -> None:
    vals = [
        log_tanh_ratio(1e9, 1.0, 1.0),
        log_tanh_ratio(1e-9, 1.0, 1.0),
        log_tanh_signed(1e9, 1.0, 1.0),
        log_tanh_signed(-1e9, 1.0, 1.0),
    ]
    assert all(-1.0 <= v <= 1.0 for v in vals)


def test_sigma_floor_applies_minimum() -> None:
    assert sigma_floor(0.0, 0.01) == 0.01
    assert sigma_floor(np.nan, 0.01) == 0.01
    assert sigma_floor(0.02, 0.01) == 0.02


def test_log_tanh_handles_pandas_na() -> None:
    """Nullable factor columns (e.g. IB-only bars) must not crash standardization."""
    assert math.isnan(log_tanh_ratio(pd.NA, 1.0, k=1.0))
    assert math.isnan(log_tanh_ratio(1.0, pd.NA, k=1.0))
    assert math.isnan(log_tanh_signed(pd.NA, 1.0, k=1.0))
    assert sigma_floor(pd.NA, 0.01) == 0.01
