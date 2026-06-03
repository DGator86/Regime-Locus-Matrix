from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_everything as re


def test_monitor_rth_only_default_on_for_master(monkeypatch) -> None:
    monkeypatch.delenv("RLM_MONITOR_RTH_ONLY", raising=False)
    assert re._monitor_rth_only_enabled(master=True) is True
    assert re._monitor_rth_only_enabled(master=False) is False


def test_monitor_rth_only_env_off(monkeypatch) -> None:
    monkeypatch.setenv("RLM_MONITOR_RTH_ONLY", "0")
    assert re._monitor_rth_only_enabled(master=True) is False


def test_pipeline_market_hours_flag_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RLM_PIPELINE_MARKET_HOURS_ONLY", "1")
    cmd = [sys.executable, "run_universe_options_pipeline.py"]
    re._extend_pipeline_cmd_from_env(cmd)
    assert "--market-hours-only" in cmd


def test_pipeline_trade_log_path_from_master(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cmd = [sys.executable, "run_universe_options_pipeline.py"]
    re._extend_pipeline_cmd_trade_log(cmd, tmp_path / "data/processed/options_large_account_trade_log.csv")
    assert "--trade-log" in cmd
    assert any("options_large_account_trade_log.csv" in str(x) for x in cmd)


def test_pipeline_short_dte_flags_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RLM_PIPELINE_SHORT_DTE", "1")
    monkeypatch.setenv("RLM_PIPELINE_DTE_MIN", "0")
    monkeypatch.setenv("RLM_PIPELINE_DTE_MAX", "5")
    cmd = [sys.executable, "run_universe_options_pipeline.py"]
    re._extend_pipeline_cmd_from_env(cmd)
    assert "--short-dte" in cmd
    assert "--dte-min" in cmd and "0" in cmd
    assert "--dte-max" in cmd and "5" in cmd
