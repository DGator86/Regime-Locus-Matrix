#!/usr/bin/env python3
"""
Startup decision-tree health — offline SPY slice + synthetic large options + PDT SPY + equity dry-run.

Uses committed files under ``data/health_snapshot/`` (see ``manifest.json``). Intended for:
  - every boot (systemd oneshot)
  - 09:00 America/New_York (invoked from ``rlm-market-hours-start.sh`` before live sync)

Does **not** call Massive/IBKR for marks; large-account SPY uses mids embedded in the snapshot plan.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SNAP = ROOT / "data" / "health_snapshot"
WORK = ROOT / "data" / "artifacts" / "startup_decision_tree_work"


def _fail(msg: str) -> None:
    print(f"[startup-health] FAIL: {msg}", flush=True)


def _ok(msg: str) -> None:
    print(f"[startup-health] OK: {msg}", flush=True)


def _prepare_workdir() -> Path:
    if WORK.exists():
        shutil.rmtree(WORK)
    for sub in ("data/raw", "data/processed"):
        (WORK / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SNAP / "bars_SPY_slice.csv", WORK / "data/raw/bars_SPY.csv")
    return WORK


def _step_core_pipeline(work: Path) -> bool:
    import pandas as pd

    from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline

    bars_path = work / "data/raw/bars_SPY.csv"
    bars = pd.read_csv(bars_path)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True)
    if bars["timestamp"].isna().all():
        _fail("bars_SPY.csv timestamps could not be parsed")
        return False
    cfg = FullRLMConfig(regime_model="hmm", hmm_states=6, use_kronos=False, attach_vix=False)
    try:
        out = FullRLMPipeline(cfg).run(bars)
    except Exception as exc:
        _fail(f"FullRLMPipeline: {exc}")
        return False
    pdf = out.policy_df
    if pdf.empty or "roee_action" not in pdf.columns:
        _fail("pipeline produced empty policy or missing roee_action")
        return False
    last = pdf.iloc[-1]
    _ok(
        f"core SPY pipeline rows={len(pdf)} last_action={last.get('roee_action')} "
        f"hmm_state={last.get('hmm_state', '?')}"
    )
    return True


def _step_large_spy_options(plan: dict) -> bool:
    from rlm.execution.dte_utils import dte_from_plan, needs_force_close
    from rlm.execution.exit_signals import EXIT_SIGNALS
    from rlm.execution.risk_targets import trailing_stop_from_peak
    from rlm.roee.chain_match import estimate_mark_value_from_matched_legs

    legs = plan.get("matched_legs") or []
    if not legs or not all("mid" in leg for leg in legs):
        _fail("large SPY snapshot plan needs matched_legs with mid on each leg")
        return False
    try:
        v = float(estimate_mark_value_from_matched_legs(legs))
    except Exception as exc:
        _fail(f"mark value from legs: {exc}")
        return False
    thr = plan.get("thresholds") or {}
    v_tp = float(thr.get("v_take_profit", float("nan")))
    v_sl = float(thr.get("v_hard_stop", float("nan")))
    v_tr_act = float(thr.get("v_trail_activate", float("nan")))
    tr_r = float(thr.get("trail_retrace_frac", 0.25))
    st: dict = {"peak_v": v, "trail_on": False}
    if v >= v_tr_act:
        st["trail_on"] = True
    plan_dte = dte_from_plan(plan)
    entry_debit = float(plan.get("entry_debit_dollars") or 0.0)
    pnl = v - entry_debit
    pnl_pct = (pnl / abs(entry_debit) * 100.0) if abs(entry_debit) > 1e-6 else float("nan")

    signal = "hold"
    force_close_dte = 0.0
    soft_time_stop_dte = 0.0
    min_profit_pct_for_soft_hold = -999.0
    max_loss_pct = -99.0
    if needs_force_close(plan, force_close_dte):
        signal = "expiry_force_close"
    elif float(soft_time_stop_dte) > 0.0 and plan_dte == plan_dte and plan_dte <= soft_time_stop_dte and (
        pnl_pct != pnl_pct or pnl_pct < float(min_profit_pct_for_soft_hold)
    ):
        signal = "time_stop"
    elif v >= v_tp:
        signal = "take_profit"
    elif v <= v_sl:
        signal = "hard_stop"
    elif st.get("trail_on"):
        peak = float(st["peak_v"])
        tstop = trailing_stop_from_peak(peak, tr_r)
        if v < tstop:
            signal = "trailing_stop"
    if pnl_pct == pnl_pct and pnl_pct <= float(max_loss_pct):
        signal = "max_loss_stop"

    _ok(f"large SPY options V={v:.2f} debit={entry_debit:.2f} signal={signal} dte={plan_dte}")
    if signal in EXIT_SIGNALS:
        _fail(f"unexpected exit signal in healthy snapshot: {signal}")
        return False
    return True


def _step_pdt_spy() -> bool:
    from rlm.challenge.models import ChallengeAccountState, PDTTracker
    from rlm.challenge.pipeline import ChallengeDecisionPipeline
    from rlm.persona.models import (
        DataStageOutput,
        GarakStageOutput,
        PersonaPipelineResult,
        SevenStageOutput,
        SiskoStageOutput,
    )

    persona = PersonaPipelineResult(
        seven=SevenStageOutput(bias="bullish", signal_alignment=0.78, confidence=0.82),  # type: ignore[arg-type]
        garak=GarakStageOutput(
            trap_risk=0.1, dealer_alignment="supportive", liquidity_comment="normal", veto=False  # type: ignore[arg-type]
        ),
        sisko=SiskoStageOutput(
            directive="long",
            entry_policy="enter on confirmation",
            invalidation_policy="close on reversal",
            target_policy="2x premium",
        ),
        data=DataStageOutput(regime_match="high", historical_edge=0.65, adaptation_note="", review_flag=False),
    )
    state = ChallengeAccountState(current_equity=1000.0)
    pdt = PDTTracker(day_trades_used_last_5d=[0])
    d = ChallengeDecisionPipeline().run("SPY", persona, state, pdt)
    if d.directive == "no_trade":
        _fail(f"PDT SPY returned no_trade: {d.reason_summary}")
        return False
    _ok(f"PDT SPY directive={d.directive} mode={d.trade_mode} conviction={d.conviction}")
    return True


def _step_equity_dry_run(work: Path) -> bool:
    proc = work / "data" / "processed"
    plan_path = SNAP / "spy_large_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    payload = {
        "generated_at_utc": "health-snapshot",
        "results": [
            plan,
            {
                "symbol": "QQQ",
                "status": "skipped",
                "skip_reason": "health_padding",
            },
        ],
    }
    plans_json = proc / "universe_trade_plans.json"
    plans_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logp = proc / "equity_health_log.csv"
    statep = proc / "equity_health_state.json"
    if not statep.exists():
        statep.write_text("{}", encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ibkr_equity_paper_trade.py"),
        "--plans",
        str(plans_json),
        "--log",
        str(logp),
        "--state",
        str(statep),
        "--position-usd",
        "1000",
        "--dry-run",
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT)})
    if r.returncode != 0:
        _fail(f"ibkr_equity_paper_trade dry-run exit {r.returncode}")
        return False
    _ok("equity leg dry-run completed")
    return True


def main() -> int:
    failures = 0
    if not (SNAP / "manifest.json").is_file():
        _fail(f"missing snapshot bundle at {SNAP}")
        return 2

    print(f"[startup-health] ROOT={ROOT} snapshot={SNAP}", flush=True)
    work = _prepare_workdir()
    _ok(f"workdir {work.relative_to(ROOT)}")

    if not _step_core_pipeline(work):
        failures += 1
    plan = json.loads((SNAP / "spy_large_plan.json").read_text(encoding="utf-8"))
    if not _step_large_spy_options(plan):
        failures += 1
    if not _step_pdt_spy():
        failures += 1
    if not _step_equity_dry_run(work):
        failures += 1

    if failures:
        print(f"[startup-health] completed with {failures} failure(s)", flush=True)
        return 1
    print("[startup-health] all decision-tree branches passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
