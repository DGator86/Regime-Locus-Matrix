"""Tests for the revamped optimization objective and regime label alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlm.optimization.base import _signal_based_score, _compute_sharpe, align_regime_labels


# ---------------------------------------------------------------------------
# _compute_sharpe
# ---------------------------------------------------------------------------

def test_compute_sharpe_positive_returns():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 252))  # positive mean, non-zero std
    s = _compute_sharpe(r)
    assert np.isfinite(s) and s > 0


def test_compute_sharpe_zero_variance():
    r = pd.Series([0.0] * 100)
    assert np.isnan(_compute_sharpe(r))


def test_compute_sharpe_too_short():
    r = pd.Series([0.01] * 5)
    assert np.isnan(_compute_sharpe(r))


# ---------------------------------------------------------------------------
# _signal_based_score
# ---------------------------------------------------------------------------

def _make_policy_df(n: int, action: str = "enter", size: float = 1.0) -> pd.DataFrame:
    prices = 100.0 * (1 + pd.Series(np.random.default_rng(0).normal(0, 0.01, n))).cumprod()
    return pd.DataFrame({
        "close": prices.values,
        "roee_action": [action] * n,
        "roee_size_fraction": [size] * n,
    })


def test_signal_based_score_returns_finite_for_entering_strategy():
    df = _make_policy_df(200, action="enter")
    score = _signal_based_score(df, oos_start=150)
    assert np.isfinite(score)


def test_signal_based_score_nan_when_oos_too_short():
    df = _make_policy_df(60, action="enter")
    score = _signal_based_score(df, oos_start=50)  # only 10 OOS bars < 30
    assert np.isnan(score)


def test_signal_based_score_nan_when_no_trades():
    df = _make_policy_df(200, action="hold")
    score = _signal_based_score(df, oos_start=150)
    # No enters → zero returns → zero Sharpe (nan from std=0) → nan score
    assert np.isnan(score)


def test_signal_based_score_transaction_costs_reduce_score():
    df = _make_policy_df(200, action="enter")
    score_no_cost = _signal_based_score(df, oos_start=150, transaction_cost_bps=0.0)
    score_high_cost = _signal_based_score(df, oos_start=150, transaction_cost_bps=0.05)
    # High costs should produce a lower (or equal) score
    if np.isfinite(score_no_cost) and np.isfinite(score_high_cost):
        assert score_high_cost <= score_no_cost + 1e-9


def test_signal_based_score_missing_close_returns_nan():
    df = pd.DataFrame({"roee_action": ["enter"] * 200, "roee_size_fraction": [1.0] * 200})
    score = _signal_based_score(df, oos_start=150)
    assert np.isnan(score)


# ---------------------------------------------------------------------------
# align_regime_labels
# ---------------------------------------------------------------------------

def test_align_regime_labels_stable_across_permutations():
    """Same economic regimes, different raw labels → same aligned output."""
    returns = np.array([0.01, 0.02, 0.05, 0.06, 0.10, 0.12])
    labels_a = np.array([0, 0, 1, 1, 2, 2])   # 0=low, 1=mid, 2=high vol
    labels_b = np.array([2, 2, 0, 0, 1, 1])   # switched: 2=low, 0=mid, 1=high vol

    aligned_a = align_regime_labels(labels_a, returns, n_regimes=3)
    aligned_b = align_regime_labels(labels_b, returns, n_regimes=3)

    np.testing.assert_array_equal(aligned_a, aligned_b)


def test_align_regime_labels_ordering():
    """Aligned state 0 has lower vol than aligned state 1."""
    rng = np.random.default_rng(7)
    low_vol = rng.normal(0, 0.005, 100)
    high_vol = rng.normal(0, 0.05, 100)
    returns = np.concatenate([low_vol, high_vol])
    raw_labels = np.array([0] * 100 + [1] * 100)  # 0=low vol (correct order)

    aligned = align_regime_labels(raw_labels, returns, n_regimes=2)

    low_vol_aligned = returns[aligned == 0]
    high_vol_aligned = returns[aligned == 1]
    assert np.std(low_vol_aligned) < np.std(high_vol_aligned)


def test_align_regime_labels_reversed_raw_order():
    """Works when raw labels are reversed: 0=high vol, 1=low vol."""
    rng = np.random.default_rng(7)
    low_vol = rng.normal(0, 0.005, 100)
    high_vol = rng.normal(0, 0.05, 100)
    returns = np.concatenate([high_vol, low_vol])
    raw_labels = np.array([0] * 100 + [1] * 100)  # 0=high vol (reversed)

    aligned = align_regime_labels(raw_labels, returns, n_regimes=2)

    low_vol_aligned = returns[aligned == 0]
    high_vol_aligned = returns[aligned == 1]
    assert np.std(low_vol_aligned) < np.std(high_vol_aligned)


def test_align_regime_labels_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        align_regime_labels(np.array([0, 1]), np.array([0.01]), n_regimes=2)
