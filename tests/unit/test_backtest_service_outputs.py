import pandas as pd

from rlm.core.pipeline import PipelineResult
from rlm.core.services.backtest_service import BacktestRequest, BacktestService


def test_backtest_service_writes_outputs(tmp_path):
    svc = BacktestService()
    req = BacktestRequest(symbol="SPY", bars_df=pd.DataFrame(), out_dir=tmp_path)
    result = PipelineResult(
        factors_df=pd.DataFrame(),
        forecast_df=pd.DataFrame(),
        policy_df=pd.DataFrame(),
        backtest_trades=pd.DataFrame({"a": [1]}),
        backtest_equity=pd.DataFrame({"e": [2]}),
        backtest_metrics={},
    )
    arts = svc.write_outputs(req, result)
    assert arts.trades_csv and arts.trades_csv.exists()
    assert arts.equity_csv and arts.equity_csv.exists()
