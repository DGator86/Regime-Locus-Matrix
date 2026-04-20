import pandas as pd

from rlm.core.pipeline import PipelineResult
from rlm.core.services.forecast_service import ForecastRequest, ForecastService


def test_forecast_service_writes_output(tmp_path):
    svc = ForecastService()
    req = ForecastRequest(symbol="SPY", bars_df=pd.DataFrame(), out_path=tmp_path / "out.csv")
    result = PipelineResult(
        factors_df=pd.DataFrame(),
        forecast_df=pd.DataFrame({"close": [1.0]}),
        policy_df=pd.DataFrame(),
    )
    arts = svc.write_outputs(req, result)
    assert arts.forecast_csv and arts.forecast_csv.is_file()
