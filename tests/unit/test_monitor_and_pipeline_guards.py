from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline
from scripts.monitor_active_trade_plans import _evaluate_plan
from scripts.run_universe_options_pipeline import (
    _apply_active_plan_guards,
    _is_infrastructure_universe_failure,
    _kronos_stub_available,
    _load_open_symbols_from_trade_log,
    _prior_universe_plans_have_actives,
    _universe_scan_is_total_infra_failure,
    should_preserve_universe_plans_on_infra_failure,
)


def _sample_plan() -> dict:
    return {
        "plan_id": "plan_1",
        "symbol": "TSLA",
        "strategy": "bull_call_spread",
        "entry_debit_dollars": 100.0,
        "entry_mid_mark_dollars": 100.0,
        "thresholds": {
            "v_take_profit": 130.0,
            "v_hard_stop": 10.0,
            "v_trail_activate": 120.0,
            "trail_retrace_frac": 0.20,
            "min_trail_exit_v": 108.0,
        },
        "matched_legs": [
            {
                "symbol": "O:TSLA260619C00100000",
                "side": "long",
                "option_type": "call",
                "strike": 100.0,
                "expiry": "2026-06-19",
                "mid": 1.0,
            }
        ],
    }


def _sample_chain(mid: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contract_symbol": "O:TSLA260619C00100000",
                "bid": mid,
                "ask": mid,
                "mid": mid,
            }
        ]
    )


def test_monitor_max_loss_stop_closes_trade(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    plan = _sample_plan()
    state: dict = {}
    monkeypatch.setattr("scripts.monitor_active_trade_plans.dte_from_plan", lambda _: 30.0)
    _evaluate_plan(
        plan,
        chain=_sample_chain(mid=0.2),  # mark=20; pnl=-80%
        state=state,
        paper_close=False,
        paper_close_dry_run=False,
        force_close_dte=14.0,
        soft_time_stop_dte=21.0,
        min_profit_pct_for_soft_hold=20.0,
        max_loss_pct=-70.0,
        min_trail_profit_frac=0.08,
        trade_log_path=log_path,
    )
    row = pd.read_csv(log_path).iloc[-1]
    assert row["signal"] == "max_loss_stop"
    assert str(row["closed"]) == "1"


def test_monitor_time_stop_and_force_close(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    plan = _sample_plan()
    state: dict = {}

    monkeypatch.setattr("scripts.monitor_active_trade_plans.dte_from_plan", lambda _: 20.0)
    monkeypatch.setattr("scripts.monitor_active_trade_plans.needs_force_close", lambda _p, _d: False)
    _evaluate_plan(
        plan,
        chain=_sample_chain(mid=1.0),  # mark=100; pnl=0%
        state=state,
        paper_close=False,
        paper_close_dry_run=False,
        force_close_dte=14.0,
        soft_time_stop_dte=21.0,
        min_profit_pct_for_soft_hold=20.0,
        max_loss_pct=-70.0,
        min_trail_profit_frac=0.08,
        trade_log_path=log_path,
    )
    first = pd.read_csv(log_path).iloc[-1]
    assert first["signal"] == "time_stop"
    assert str(first["closed"]) == "1"

    monkeypatch.setattr("scripts.monitor_active_trade_plans.dte_from_plan", lambda _: 13.0)
    monkeypatch.setattr("scripts.monitor_active_trade_plans.needs_force_close", lambda _p, _d: True)
    _evaluate_plan(
        plan,
        chain=_sample_chain(mid=1.5),  # any PnL should still force close
        state=state,
        paper_close=False,
        paper_close_dry_run=False,
        force_close_dte=14.0,
        soft_time_stop_dte=21.0,
        min_profit_pct_for_soft_hold=20.0,
        max_loss_pct=-70.0,
        min_trail_profit_frac=0.08,
        trade_log_path=log_path,
    )
    second = pd.read_csv(log_path).iloc[-1]
    assert second["signal"] == "expiry_force_close"
    assert str(second["closed"]) == "1"


def test_monitor_trailing_stop_holds_when_below_profit_floor(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    plan = _sample_plan()
    state = {"plan_1": {"peak_v": 130.0, "trail_on": True}}
    monkeypatch.setattr("scripts.monitor_active_trade_plans.dte_from_plan", lambda _: 30.0)
    _evaluate_plan(
        plan,
        chain=_sample_chain(mid=0.9),  # mark=90 < trail stop from peak but < min_trail_exit_v 108
        state=state,
        paper_close=False,
        paper_close_dry_run=False,
        force_close_dte=0.0,
        soft_time_stop_dte=0.0,
        min_profit_pct_for_soft_hold=20.0,
        max_loss_pct=-70.0,
        min_trail_profit_frac=0.08,
        trade_log_path=log_path,
    )
    row = pd.read_csv(log_path).iloc[-1]
    assert row["signal"] == "hold"
    assert str(row["closed"]) == "0"


def test_monitor_default_lifecycle_stops_do_not_close_fresh_trade(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    plan = _sample_plan()
    state: dict = {}

    monkeypatch.setattr("scripts.monitor_active_trade_plans.dte_from_plan", lambda _: 20.0)
    _evaluate_plan(
        plan,
        chain=_sample_chain(mid=1.0),  # mark=100; pnl=0%
        state=state,
        paper_close=False,
        paper_close_dry_run=False,
        force_close_dte=0.0,
        soft_time_stop_dte=0.0,
        min_profit_pct_for_soft_hold=20.0,
        max_loss_pct=-70.0,
        min_trail_profit_frac=0.08,
        trade_log_path=log_path,
    )

    row = pd.read_csv(log_path).iloc[-1]
    assert row["signal"] == "hold"
    assert str(row["closed"]) == "0"


def test_pipeline_duplicate_symbol_trimmed() -> None:
    rows = [
        {"symbol": "TSLA", "status": "active", "rank_score": 0.9, "strategy": "a"},
        {"symbol": "TSLA", "status": "active", "rank_score": 0.8, "strategy": "b"},
        {"symbol": "AAPL", "status": "active", "rank_score": 0.7, "strategy": "a"},
    ]
    _apply_active_plan_guards(rows, max_active_per_symbol=1, open_symbols=set())
    tsla_active = [r for r in rows if r.get("symbol") == "TSLA" and r.get("status") == "active"]
    assert len(tsla_active) == 1
    trimmed = [r for r in rows if r.get("symbol") == "TSLA" and r.get("status") == "trimmed"]
    assert trimmed and trimmed[0].get("skip_reason") == "duplicate_symbol_strategy_or_max_active_per_symbol"


def test_infrastructure_failure_classifier() -> None:
    assert _is_infrastructure_universe_failure(
        {"status": "error", "skip_reason": "ConnectionError: Massive"}
    )
    assert _is_infrastructure_universe_failure({"status": "skipped", "skip_reason": "no_stock_bars"})
    assert _is_infrastructure_universe_failure(
        {"status": "skipped", "skip_reason": "massive_chain_error:timeout"}
    )
    assert _is_infrastructure_universe_failure(
        {"status": "skipped", "skip_reason": "outside_entry_window (pre-market)"}
    )
    assert not _is_infrastructure_universe_failure(
        {"status": "skipped", "skip_reason": "options_edge_gate"}
    )
    assert not _is_infrastructure_universe_failure(
        {"status": "skipped", "skip_reason": "roee_skip_or_no_legs"}
    )
    assert not _is_infrastructure_universe_failure({"status": "active", "skip_reason": ""})


def test_total_infra_failure_preserves_prior_actives_heuristic(tmp_path: Path) -> None:
    failed = [
        {"symbol": "SPY", "status": "error", "skip_reason": "Timeout"},
        {"symbol": "QQQ", "status": "skipped", "skip_reason": "no_stock_bars"},
        {"symbol": "IWM", "status": "skipped", "skip_reason": "massive_chain_error:missing_chain"},
    ]
    legit_empty = [
        {"symbol": "SPY", "status": "skipped", "skip_reason": "options_edge_gate"},
        {"symbol": "QQQ", "status": "skipped", "skip_reason": "timeframe_confirmation_block"},
    ]
    assert _universe_scan_is_total_infra_failure(failed)
    assert not _universe_scan_is_total_infra_failure(legit_empty)
    assert not _universe_scan_is_total_infra_failure(
        failed + [{"symbol": "AAPL", "status": "skipped", "skip_reason": "roee_skip_or_no_legs"}]
    )

    plans = tmp_path / "universe_trade_plans.json"
    assert not _prior_universe_plans_have_actives(plans)
    plans.write_text(
        '{"active_ranked":[{"status":"active","plan_id":"p1","symbol":"SPY"}],'
        '"results":[{"status":"active","plan_id":"p1","symbol":"SPY"}]}',
        encoding="utf-8",
    )
    assert _prior_universe_plans_have_actives(plans)
    assert should_preserve_universe_plans_on_infra_failure(
        final_active=[],
        final_results=failed,
        out_path=plans,
    )
    assert not should_preserve_universe_plans_on_infra_failure(
        final_active=[],
        final_results=legit_empty,
        out_path=plans,
    )
    assert not should_preserve_universe_plans_on_infra_failure(
        final_active=[{"status": "active", "plan_id": "n1"}],
        final_results=failed,
        out_path=plans,
    )
    prior = plans.read_text(encoding="utf-8")
    # Publishing must not run when preserve=True (caller skips write).
    assert "p1" in prior


def test_pipeline_open_symbol_in_trade_log_blocks_new_active(tmp_path: Path) -> None:
    trade_log = tmp_path / "trade_log.csv"
    trade_log.write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
        "2026-04-28T00:00:00Z,old_tsla,TSLA,x,1,1,1,1,0,0,hold,0,20\n",
        encoding="utf-8",
    )
    open_symbols = _load_open_symbols_from_trade_log(trade_log)
    assert "TSLA" in open_symbols

    rows = [{"symbol": "TSLA", "status": "active", "rank_score": 0.9, "strategy": "a"}]
    _apply_active_plan_guards(rows, max_active_per_symbol=1, open_symbols=open_symbols)
    assert rows[0]["status"] == "trimmed"
    assert rows[0]["skip_reason"] == "symbol_already_open_in_trade_log"


def test_full_pipeline_nightly_overlay_ignores_stale_mtf_regimes(tmp_path: Path) -> None:
    nightly_path = tmp_path / "live_nightly_hyperparams.json"
    nightly_path.write_text('{"mtf_regimes": true, "move_window": 91}', encoding="utf-8")

    cfg = FullRLMConfig(nightly_hyperparams_path=str(nightly_path))
    pipe = FullRLMPipeline(cfg)

    assert pipe.config.move_window == 91
    assert pipe.config.mtf_regimes is False


def test_kronos_stub_detector_returns_bool() -> None:
    assert isinstance(_kronos_stub_available(), bool)
