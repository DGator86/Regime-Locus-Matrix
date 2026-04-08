import numpy as np
import pandas as pd

from rlm.backtest.walkforward import WalkForwardConfig, run_walkforward
from rlm.types.forecast import ForecastConfig


def make_bars(n: int = 220) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    trend = np.linspace(5000, 5100, n)
    wave = np.sin(np.arange(n) / 7.0) * 8.0
    return pd.DataFrame(
        {
            "open": trend + wave - 1.0,
            "high": trend + wave + 2.0,
            "low": trend + wave - 2.0,
            "close": trend + wave,
            "volume": 100000 + (np.arange(n) % 10) * 5000,
            "vwap": trend + wave * 0.25,
            "anchored_vwap": trend - 2.0,
            "buy_volume": 60000 + (np.arange(n) % 5) * 1500,
            "sell_volume": 45000 + (np.arange(n) % 6) * 1400,
            "advancers": 1700 + (np.arange(n) % 20) * 10,
            "decliners": 1300 + (np.arange(n) % 15) * 10,
            "index_return": pd.Series(trend).pct_change(10).fillna(0.0).values,
            "vix": 17 + (np.arange(n) % 15) * 0.4,
            "vvix": 90 + (np.arange(n) % 12) * 1.0,
            "bid_ask_spread": 0.5 + (np.arange(n) % 5) * 0.02,
            "order_book_depth": 2500 + (np.arange(n) % 7) * 100,
            "gex": np.sin(np.arange(n) / 10.0),
            "vanna": np.cos(np.arange(n) / 12.0),
            "charm": np.sin(np.arange(n) / 9.0),
            "put_call_skew": 0.02 + (np.arange(n) % 5) * 0.005,
            "iv_rank": 0.45 + (np.arange(n) % 10) * 0.02,
            "term_structure_ratio": 0.95 + (np.arange(n) % 5) * 0.02,
            "dealer_position_proxy": np.sin(np.arange(n) / 16.0) * 0.2,
        },
        index=idx,
    )


def make_chain(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expiries = [pd.Timestamp("2025-02-14"), pd.Timestamp("2025-03-21")]
    for ts, row in bars.iterrows():
        px = float(row["close"])
        base_strikes = [round(px / 5) * 5 + k for k in range(-50, 55, 25)]
        for expiry in expiries:
            for option_type in ["call", "put"]:
                for strike in base_strikes:
                    rows.append(
                        {
                            "timestamp": ts,
                            "underlying": "SPY",
                            "expiry": expiry,
                            "option_type": option_type,
                            "strike": float(strike),
                            "bid": 8.0,
                            "ask": 8.4,
                        }
                    )
    return pd.DataFrame(rows)


def test_run_walkforward_executes() -> None:
    bars = make_bars()
    chain = make_chain(bars)

    equity_df, trades_df, summary_df = run_walkforward(
        bars=bars,
        option_chain=chain,
        forecast_config=ForecastConfig(),
        wf_config=WalkForwardConfig(
            is_window=100,
            oos_window=50,
            step_size=50,
            initial_capital=100000.0,
            strike_increment=5.0,
            underlying_symbol="SPY",
            quantity_per_trade=1,
        ),
    )

    assert summary_df is not None
    assert len(summary_df) > 0


def test_run_walkforward_reports_purge_and_regime_metadata() -> None:
    bars = make_bars(260)
    chain = make_chain(bars)

    _, _, summary_df = run_walkforward(
        bars=bars,
        option_chain=chain,
        forecast_config=ForecastConfig(),
        wf_config=WalkForwardConfig(
            is_window=100,
            oos_window=40,
            step_size=40,
            purge_bars=5,
            regime_aware=True,
            min_regime_train_samples=5,
        ),
    )

    assert not summary_df.empty
    assert "purge_bars" in summary_df.columns
    assert "regime_aware" in summary_df.columns
    assert "coverage_adjusted" in summary_df.columns
    assert "unsafe_oos_bars" in summary_df.columns
    assert "last_oos_regime_train_samples" in summary_df.columns
    assert "min_oos_regime_train_samples" in summary_df.columns
    assert "regime_safety_passed" in summary_df.columns
    assert int(summary_df["purge_bars"].iloc[0]) == 5
    assert bool(summary_df["regime_aware"].iloc[0]) is True