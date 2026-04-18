from __future__ import annotations

import pandas as pd

from rlm.training.datasets import build_regime_training_frame, build_strategy_value_training_frame


def _base_df(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": f"2026-01-01T00:{i:02d}:00Z",
                "symbol": "SPY",
                "close": 100.0 + i,
                "sigma": 0.01 + i * 1e-4,
                "M_D": 5.0 + (i % 3),
                "M_V": 5.0 - (i % 2),
                "M_L": 5.0 + (i % 4) * 0.1,
                "M_G": 5.0 + (i % 5) * 0.2,
                "M_trend_strength": float(i % 4),
                "M_dealer_control": float(i % 3),
                "M_alignment": float((i % 3) - 1),
                "M_delta_neutral": float(i % 6),
                "M_R_trans": float(i % 2),
            }
        )
    return pd.DataFrame(rows)


def test_dataset_builders_produce_expected_columns() -> None:
    df = _base_df()
    regime = build_regime_training_frame(df)
    value = build_strategy_value_training_frame(df, horizon=5)

    assert "regime_label" in regime.columns
    for col in (
        "bull_call_spread",
        "bear_put_spread",
        "iron_condor",
        "calendar_spread",
        "debit_spread",
        "no_trade",
    ):
        assert col in value.columns


def test_no_trade_target_is_always_zero() -> None:
    value = build_strategy_value_training_frame(_base_df(), horizon=4)
    assert (value["no_trade"] == 0.0).all()


def test_regime_labels_generated_for_valid_rows() -> None:
    regime = build_regime_training_frame(_base_df())
    assert regime["regime_label"].notna().all()
    assert regime["regime_label"].str.len().gt(0).all()
