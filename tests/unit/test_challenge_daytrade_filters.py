"""Unit tests for challenge intraday sniper filters."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from rlm.challenge.daytrade_filters import is_great_daytrade_setup


def _bullish_intraday_frame() -> pd.DataFrame:
    rows = []
    for i in range(19):
        rows.append({"high": 100.2, "low": 99.8, "close": 100.0, "volume": 100.0})
    rows.append({"high": 103.0, "low": 101.5, "close": 102.5, "volume": 400.0})
    return pd.DataFrame(rows)


def test_sniper_filter_returns_false_for_missing_ohlcv_schema() -> None:
    """Supplying partial intraday data should reject the setup, not crash."""
    intraday = pd.DataFrame({"close": [100.0] * 20})
    current_bar = {"close": 100.0}

    assert (
        is_great_daytrade_setup(
            "SPY",
            ("bull", "low_vol", "high_liquidity", "supportive"),
            current_bar,  # type: ignore[arg-type]
            intraday,
        )
        is False
    )


def test_sniper_filter_accepts_case_variant_ohlcv_columns() -> None:
    """Common OHLCV case variants should work with the same gate logic."""
    intraday = _bullish_intraday_frame().rename(
        columns={"high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    current_bar = pd.Series({"High": 103.0, "Low": 101.5, "Close": 102.5, "Volume": 400.0})

    with patch("rlm.challenge.daytrade_filters.get_iv_rank", return_value=20.0):
        assert (
            is_great_daytrade_setup(
                "SPY",
                ("bull", "low_vol", "high_liquidity", "supportive"),
                current_bar,
                intraday,
            )
            is True
        )
