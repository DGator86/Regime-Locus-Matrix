import pandas as pd

from rlm.core.services.backtest_service import _aggregate_walkforward_metrics


def test_aggregate_walkforward_metrics_averages() -> None:
    df = pd.DataFrame(
        {
            "sharpe": [1.0, 2.0],
            "total_return_pct": [0.01, 0.02],
            "max_drawdown": [-0.1, -0.2],
            "num_trades": [3.0, 5.0],
        }
    )
    out = _aggregate_walkforward_metrics(df)
    assert out["wf_windows"] == 2.0
    assert out["wf_mean_sharpe"] == 1.5
    assert out["wf_mean_total_return_pct"] == 0.015


def test_aggregate_walkforward_metrics_empty() -> None:
    assert _aggregate_walkforward_metrics(pd.DataFrame()) == {}
