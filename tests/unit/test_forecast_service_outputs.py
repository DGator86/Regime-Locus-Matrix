"""Unit tests for ForecastService — artifact writing and summarize contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline
from rlm.core.services.forecast_service import ForecastRequest, ForecastService
from rlm.data.synthetic import synthetic_bars_demo


@pytest.fixture(scope="module")
def bars_df() -> pd.DataFrame:
    return synthetic_bars_demo(end=pd.Timestamp("2024-12-31"), periods=200)


@pytest.fixture(scope="module")
def forecast_result(bars_df: pd.DataFrame):
    cfg = FullRLMConfig(
        symbol="TEST",
        regime_model="hmm",
        hmm_states=4,
        use_kronos=False,
        attach_vix=False,
        run_backtest=False,
    )
    return FullRLMPipeline(cfg).run(bars_df)


class TestForecastServiceRun:
    def test_run_returns_pipeline_result(self, bars_df: pd.DataFrame) -> None:
        req = ForecastRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(
                symbol="TEST",
                regime_model="hmm",
                hmm_states=4,
                use_kronos=False,
                attach_vix=False,
            ),
        )
        result = ForecastService().run(req)
        assert not result.forecast_df.empty
        assert not result.policy_df.empty

    def test_run_does_not_require_option_chain(self, bars_df: pd.DataFrame) -> None:
        req = ForecastRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(
                symbol="TEST", use_kronos=False, attach_vix=False
            ),
        )
        result = ForecastService().run(req)
        assert result is not None


class TestForecastServiceWriteOutputs:
    def test_write_outputs_creates_csv(self, tmp_path: Path, bars_df: pd.DataFrame) -> None:
        out = tmp_path / "forecast_TEST.csv"
        req = ForecastRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(symbol="TEST", use_kronos=False, attach_vix=False),
            out_path=out,
            write_output=True,
            data_root=str(tmp_path),
        )
        result = ForecastService().run(req)
        arts = ForecastService().write_outputs(req, result)
        assert arts.forecast_csv == out
        assert out.is_file()
        assert arts.rows_written > 0

    def test_write_outputs_skipped_when_disabled(self, tmp_path: Path, bars_df: pd.DataFrame) -> None:
        req = ForecastRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(symbol="TEST", use_kronos=False, attach_vix=False),
            write_output=False,
            data_root=str(tmp_path),
        )
        result = ForecastService().run(req)
        arts = ForecastService().write_outputs(req, result)
        assert arts.forecast_csv is None

    def test_write_outputs_creates_manifest(self, tmp_path: Path, bars_df: pd.DataFrame) -> None:
        req = ForecastRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(symbol="TEST", use_kronos=False, attach_vix=False),
            write_output=False,
            data_root=str(tmp_path),
        )
        result = ForecastService().run(req)
        ForecastService().write_outputs(req, result)
        runs_dir = tmp_path / "artifacts" / "runs"
        assert runs_dir.is_dir()
        manifests = list(runs_dir.glob("*.json"))
        assert len(manifests) == 1


class TestForecastServiceSummarize:
    def test_summarize_returns_dict(self, forecast_result) -> None:
        summary = ForecastService().summarize(forecast_result)
        assert isinstance(summary, dict)

    def test_summarize_has_rows(self, forecast_result) -> None:
        summary = ForecastService().summarize(forecast_result)
        assert "rows" in summary
        assert summary["rows"] > 0

    def test_empty_forecast_returns_empty_dict(self) -> None:
        from rlm.core.pipeline import PipelineResult
        import pandas as pd
        result = PipelineResult(
            factors_df=pd.DataFrame(),
            forecast_df=pd.DataFrame(),
            policy_df=pd.DataFrame(),
        )
        summary = ForecastService().summarize(result)
        assert summary == {}
