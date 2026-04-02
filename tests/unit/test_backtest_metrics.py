import pandas as pd

from rlm.backtest.metrics import summarize_backtest


def test_summarize_backtest_runs() -> None:
    equity = pd.DataFrame(
        {
            "equity": [100000, 101000, 100500, 102000],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="D"),
    )

    trades = pd.DataFrame(
        {
            "pnl": [100, -50, 200],
            "pnl_pct": [0.10, -0.05, 0.15],
        }
    )

    summary = summarize_backtest(equity, trades)
    assert "final_equity" in summary
    assert "max_drawdown" in summary
    assert "sharpe" in summary
    assert "win_rate" in summary
