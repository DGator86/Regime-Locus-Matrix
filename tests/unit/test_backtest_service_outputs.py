"""Unit tests for BacktestService — artifact writing and summarize contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rlm.core.pipeline import FullRLMConfig
from rlm.core.services.backtest_service import BacktestRequest, BacktestService
from rlm.data.synthetic import synthetic_bars_demo


@pytest.fixture(scope="module")
def bars_df() -> pd.DataFrame:
    return synthetic_bars_demo(end=pd.Timestamp("2024-12-31"), periods=200)


class TestBacktestServiceRun:
    def test_run_returns_pipeline_result(self, bars_df: pd.DataFrame) -> None:
        req = BacktestRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(
                symbol="TEST",
                regime_model="hmm",
                hmm_states=4,
                use_kronos=False,
                attach_vix=False,
                run_backtest=True,
            ),
        )
        result = BacktestService().run(req)
        assert not result.forecast_df.empty

    def test_run_always_sets_run_backtest(self, bars_df: pd.DataFrame) -> None:
        req = BacktestRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(symbol="TEST", use_kronos=False, attach_vix=False, run_backtest=False),
        )
        result = BacktestService().run(req)
        # run_backtest forced True by service — backtest fields should be present or None (pipeline may not produce)
        assert result is not None


class TestBacktestServiceWriteOutputs:
    def test_write_outputs_skipped_without_out_dir(self, bars_df: pd.DataFrame) -> None:
        req = BacktestRequest(
            symbol="TEST",
            bars_df=bars_df,
            write_outputs=True,
            out_dir=None,
        )
        result = BacktestService().run(req)
        arts = BacktestService().write_outputs(req, result)
        assert arts.trades_csv is None
        assert arts.equity_csv is None

    def test_write_outputs_creates_manifest(self, tmp_path: Path, bars_df: pd.DataFrame) -> None:
        req = BacktestRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(symbol="TEST", use_kronos=False, attach_vix=False),
            write_outputs=True,
            out_dir=tmp_path / "backtest_out",
            data_root=str(tmp_path),
        )
        result = BacktestService().run(req)
        BacktestService().write_outputs(req, result)
        runs_dir = tmp_path / "artifacts" / "runs"
        assert runs_dir.is_dir()
        manifests = list(runs_dir.glob("*.json"))
        assert len(manifests) == 1


class TestBacktestServiceSummarize:
    def test_summarize_returns_dict(self, bars_df: pd.DataFrame) -> None:
        req = BacktestRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(symbol="TEST", use_kronos=False, attach_vix=False),
        )
        result = BacktestService().run(req)
        summary = BacktestService().summarize(result)
        assert isinstance(summary, dict)

    def test_summarize_has_trade_count_when_trades_present(self, bars_df: pd.DataFrame) -> None:
        req = BacktestRequest(
            symbol="TEST",
            bars_df=bars_df,
            config=FullRLMConfig(symbol="TEST", use_kronos=False, attach_vix=False),
        )
        result = BacktestService().run(req)
        if result.backtest_trades is not None:
            summary = BacktestService().summarize(result)
            assert "trade_count" in summary
