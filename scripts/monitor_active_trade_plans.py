#!/usr/bin/env python3
"""
Poll **Massive** option snapshots for open trade plans and evaluate **mid-mark**
vs exit rules:

- **Take profit** when ``V >= v_take_profit``
- **Hard stop** when ``V <= v_hard_stop``
- **Trailing stop** after ``V`` reaches ``v_trail_activate``: maintain ``peak``; exit if
  ``V < peak * (1 - trail_retrace_frac)``

Input: JSON from ``scripts/run_universe_options_pipeline.py`` (``--out``).

State file (repo root relative): ``data/processed/trade_monitor_state.json`` tracks ``peak_v`` and
``trail_on`` per ``plan_id``.  ``trade_plan_snapshots.json`` (same directory) stores the last known
``matched_legs`` / combo for each evaluated plan so Telegram ``/positions`` still formats rows
after a plan drops out of ``universe_trade_plans.json``.

**IBKR policy:** Do not use ``--paper-close`` for live orders — it is **disabled** (exit). Use
``--paper-close-dry-run`` to log close intent only. IBKR in this project is **equities** only.

Examples::

    python scripts/monitor_active_trade_plans.py \
        --plans data/processed/universe_trade_plans.json --once
    python scripts/monitor_active_trade_plans.py \
        --plans data/processed/universe_trade_plans.json --interval 120
    python scripts/monitor_active_trade_plans.py \
        --plans data/processed/universe_trade_plans.json --paper-close-dry-run --once
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("RLM_ROOT", str(REPO_ROOT))).expanduser().resolve()
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# ruff: noqa: E402
from rlm.data.massive import MassiveClient
from rlm.data.massive_option_chain import massive_option_chains_from_client
from rlm.execution.dte_utils import dte_from_plan, needs_force_close
from rlm.execution.exit_signals import EXIT_SIGNALS
from rlm.execution.combo_spec import (
    legs_from_combo_spec,
    plan_combo_spec,
    reverse_combo_legs,
)
from rlm.execution.risk_targets import trailing_stop_from_peak
from rlm.roee.chain_match import estimate_mark_value_from_matched_legs
from rlm.utils.market_hours import is_rth_now, session_label


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


_TRADE_LOG_COLUMNS = [
    "timestamp_utc",
    "plan_id",
    "symbol",
    "strategy",
    "entry_debit",
    "entry_mid",
    "current_mark",
    "peak_mark",
    "unrealized_pnl",
    "unrealized_pnl_pct",
    "signal",
    "closed",
    "dte",
    "legs_json",
]


def _migrate_trade_log_add_columns(log_path: Path, fieldnames: list[str]) -> None:
    """Rewrite CSV when new monitor columns appear (e.g. legs_json)."""
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return
    try:
        with log_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            old_names = list(reader.fieldnames or [])
            if old_names and all(c in old_names for c in fieldnames):
                return
            rows = list(reader)
    except OSError:
        return
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (r.get(k) or "") for k in fieldnames})


def _legs_json_for_trade_log(updated: list[dict] | None) -> str:
    """Compact legs for CSV (Telegram + bookkeeping without universe JSON)."""
    if not updated:
        return ""
    slim: list[dict[str, object]] = []
    for m in updated:
        exp = m.get("expiry")
        exp_s = str(pd.Timestamp(exp).date()) if exp is not None else str(m.get("expiry") or "")
        slim.append(
            {
                "side": m.get("side"),
                "option_type": m.get("option_type"),
                "strike": m.get("strike"),
                "expiry": exp_s[:10] if exp_s else "",
            }
        )
    return json.dumps(slim, separators=(",", ":"), default=str)


def _append_trade_log(log_path: Path, row: dict) -> None:
    """Upsert one row per ``plan_id`` (latest state) to avoid multi-GB poll append logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.is_file() and log_path.stat().st_size > 0:
        _migrate_trade_log_add_columns(log_path, _TRADE_LOG_COLUMNS)
    rows: dict[str, dict[str, str]] = {}
    if log_path.is_file() and log_path.stat().st_size > 0:
        try:
            with log_path.open("r", encoding="utf-8", newline="") as f:
                for existing in csv.DictReader(f):
                    pid = str(existing.get("plan_id") or "").strip()
                    if pid:
                        rows[pid] = {k: str(existing.get(k, "")) for k in _TRADE_LOG_COLUMNS}
        except OSError:
            rows = {}
    pid = str(row.get("plan_id") or "").strip()
    if pid:
        merged = rows.get(pid, {})
        for col in _TRADE_LOG_COLUMNS:
            if col in row and row[col] is not None:
                merged[col] = str(row[col])
        rows[pid] = merged
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TRADE_LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows.values():
            writer.writerow(r)


def _contract_key(leg: dict) -> str:
    return str(leg.get("symbol") or leg.get("contract_symbol") or "")


def _parse_symbols(s: str) -> list[str]:
    parts = [x.strip().upper() for x in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _build_incremental_snapshot_params(plans: list[dict], *, massive_limit: int) -> dict[str, object]:
    params: dict[str, object] = {
        "limit": int(massive_limit),
        "sort": "expiration_date",
        "order": "asc",
    }
    expiries: list[str] = []
    strikes: list[float] = []
    option_types: set[str] = set()
    for plan in plans:
        for leg in plan.get("matched_legs") or []:
            expiry = leg.get("expiry")
            if expiry:
                expiries.append(str(pd.Timestamp(expiry).date()))
            strike = leg.get("strike")
            if strike is not None:
                try:
                    strikes.append(float(strike))
                except (TypeError, ValueError):
                    pass
            option_type = str(leg.get("option_type", "")).lower().strip()
            if option_type in {"call", "put"}:
                option_types.add(option_type)
    if expiries:
        params["expiration_date.gte"] = min(expiries)
        params["expiration_date.lte"] = max(expiries)
    if strikes:
        params["strike_price.gte"] = min(strikes)
        params["strike_price.lte"] = max(strikes)
    if len(option_types) == 1:
        params["contract_type"] = next(iter(option_types))
    return params


def _refresh_matched_mids(chain: pd.DataFrame, matched_legs: list[dict]) -> list[dict] | None:
    if chain.empty or not matched_legs:
        return None
    sym_col = "contract_symbol" if "contract_symbol" in chain.columns else None
    if sym_col is None:
        return None

    updated: list[dict] = []
    for m in matched_legs:
        key = _contract_key(m)
        if not key:
            return None
        sub = chain[chain[sym_col].astype(str) == key]
        if sub.empty:
            return None
        r = sub.iloc[0]
        bid = float(r["bid"])
        ask = float(r["ask"])
        mid = float(r["mid"]) if "mid" in r and pd.notna(r["mid"]) else (bid + ask) / 2.0
        u = dict(m)
        u["bid"] = bid
        u["ask"] = ask
        u["mid"] = mid
        updated.append(u)
    return updated


def _persist_trade_plan_snapshot(plan_snapshots: dict[str, dict], plan: dict) -> None:
    """Remember leg structure + risk marks for open rows after the plan leaves universe JSON."""
    pid = str(plan.get("plan_id") or "").strip()
    if not pid:
        return
    spec = plan_combo_spec(plan)
    plan_snapshots[pid] = {
        "plan_id": plan.get("plan_id"),
        "symbol": plan.get("symbol"),
        "strategy": plan.get("strategy"),
        "decision": plan.get("decision"),
        "matched_legs": list(plan.get("matched_legs") or []),
        "combo_spec": spec,
        "candidate": plan.get("candidate"),
        "regime_key": plan.get("regime_key"),
        "thresholds": plan.get("thresholds") or {},
        "entry_debit_dollars": plan.get("entry_debit_dollars"),
        "entry_mid_mark_dollars": plan.get("entry_mid_mark_dollars"),
    }


def _open_trade_log_plan_ids(log_path: Path) -> set[str]:
    """``plan_id`` values whose latest CSV row is still open (``closed`` != 1)."""
    if not log_path.is_file():
        return set()
    latest: dict[str, dict[str, str]] = {}
    try:
        with log_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("plan_id") or "").strip()
                if pid:
                    latest[pid] = {k: str(v) for k, v in row.items()}
    except OSError:
        return set()
    return {pid for pid, row in latest.items() if (row.get("closed") or "0").strip() != "1"}


def _evaluate_plan(
    plan: dict,
    *,
    chain: pd.DataFrame,
    state: dict,
    paper_close: bool,
    paper_close_dry_run: bool,
    force_close_dte: float,
    soft_time_stop_dte: float,
    min_profit_pct_for_soft_hold: float,
    max_loss_pct: float,
    trade_log_path: Path | None = None,
    plan_snapshots: dict[str, dict] | None = None,
) -> None:
    pid = str(plan.get("plan_id") or plan.get("symbol") or "unknown")
    thr = plan.get("thresholds") or {}
    mlegs = plan.get("matched_legs") or []
    sym = str(plan.get("symbol", ""))

    if plan_snapshots is not None:
        _persist_trade_plan_snapshot(plan_snapshots, plan)

    updated = _refresh_matched_mids(chain, mlegs)
    if updated is None:
        print(f"[{sym}] {pid} SKIP refresh (contracts missing from snapshot)")
        return

    v = float(estimate_mark_value_from_matched_legs(updated))
    v_tp = float(thr.get("v_take_profit", float("nan")))
    v_sl = float(thr.get("v_hard_stop", float("nan")))
    v_tr_act = float(thr.get("v_trail_activate", float("nan")))
    tr_r = float(thr.get("trail_retrace_frac", 0.25))

    st = state.setdefault(pid, {"peak_v": v, "trail_on": False})

    if v > float(st.get("peak_v", v)):
        st["peak_v"] = v

    if not st.get("trail_on") and v >= v_tr_act:
        st["trail_on"] = True
        print(f"[{sym}] {pid} TRAIL ARMED  V={v:.2f} >= activate={v_tr_act:.2f}")

    # DTE-based force close: emit before TP/stop to ensure 0DTE positions are closed
    plan_dte = dte_from_plan(plan)
    dte_suffix = f"  DTE={plan_dte:.3f}" if plan_dte == plan_dte else ""
    msg = (
        f"[{sym}] {pid} V={v:.2f}  "
        f"V0={plan.get('entry_mid_mark_dollars')}  "
        f"debit={plan.get('entry_debit_dollars')}{dte_suffix}"
    )

    entry_debit = float(plan.get("entry_debit_dollars") or 0.0)
    entry_mid = float(plan.get("entry_mid_mark_dollars") or 0.0)
    # Use entry_debit as the cost basis (positive = paid a debit, negative = received credit).
    pnl = v - entry_debit
    pnl_pct = (pnl / abs(entry_debit) * 100.0) if abs(entry_debit) > 1e-6 else float("nan")

    signal = "hold"
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

    if signal == "expiry_force_close":
        print(f"{msg}  ACTION: EXPIRY_FORCE_CLOSE (DTE={plan_dte:.3f} <= {force_close_dte})")
    elif signal == "take_profit":
        print(f"{msg}  ACTION: TAKE_PROFIT (V >= {v_tp:.2f})")
    elif signal == "hard_stop":
        print(f"{msg}  ACTION: HARD_STOP (V <= {v_sl:.2f})")
    elif signal == "max_loss_stop":
        print(f"{msg}  ACTION: MAX_LOSS_STOP (PnL={pnl_pct:.2f}% <= {max_loss_pct:.2f}%)")
    elif signal == "time_stop":
        print(
            f"{msg}  ACTION: TIME_STOP (DTE={plan_dte:.3f} <= {soft_time_stop_dte}"
            f" and PnL={pnl_pct:.2f}% < {min_profit_pct_for_soft_hold:.2f}%)"
        )
    elif signal == "trailing_stop":
        peak = float(st["peak_v"])
        tstop = trailing_stop_from_peak(peak, tr_r)
        print(f"{msg}  ACTION: TRAILING_STOP (V < {tstop:.2f} peak={peak:.2f})")
    elif st.get("trail_on"):
        peak = float(st["peak_v"])
        tstop = trailing_stop_from_peak(peak, tr_r)
        print(f"{msg}  ok  trail_peak={peak:.2f} trail_stop={tstop:.2f}")
    else:
        print(f"{msg}  ok  (trail not armed)")

    st["last_signal"] = signal

    if plan_snapshots is not None and updated is not None:
        _persist_trade_plan_snapshot(plan_snapshots, {**plan, "matched_legs": list(updated)})

    # --- Trade log -----------------------------------------------------------
    if trade_log_path is not None:
        _append_trade_log(
            trade_log_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "plan_id": pid,
                "symbol": sym,
                "strategy": str(plan.get("strategy") or ""),
                "entry_debit": round(entry_debit, 4),
                "entry_mid": round(entry_mid, 4),
                "current_mark": round(v, 4),
                "peak_mark": round(float(st.get("peak_v", v)), 4),
                "unrealized_pnl": round(pnl, 4),
                "unrealized_pnl_pct": round(pnl_pct, 2) if pnl_pct == pnl_pct else "",
                "signal": signal,
                "closed": "1" if signal in EXIT_SIGNALS else "0",
                "dte": round(plan_dte, 3) if plan_dte == plan_dte else "",
                "legs_json": _legs_json_for_trade_log(updated),
            },
        )
    # -------------------------------------------------------------------------

    if (
        paper_close_dry_run
        and signal in EXIT_SIGNALS
        and not st.get("paper_close_sent")
        and plan_combo_spec(plan) is not None
    ):
        spec = plan_combo_spec(plan)
        assert spec is not None
        qty = int(spec.get("quantity", 1))
        try:
            open_legs = legs_from_combo_spec(spec)
            close_legs = reverse_combo_legs(open_legs)
        except Exception as e:
            print(f"[{sym}] {pid} PAPER-CLOSE skip (bad spec): {e}", file=sys.stderr)
            return

        oa = str(spec.get("combo_order_action", "BUY")).upper()
        close_parent = "BUY" if oa == "SELL" else "SELL"

        print(f"[{sym}] {pid} PAPER-CLOSE DRY-RUN MKT " f"{close_parent} qty={qty} legs={len(close_legs)}")
        st["paper_close_sent"] = True


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--plans", type=Path, required=True, help="universe_trade_plans.json")
    p.add_argument(
        "--state",
        type=Path,
        default=Path("data/processed/trade_monitor_state.json"),
        help="Persistent peak/trail state",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between polls (if not --once)",
    )
    p.add_argument("--once", action="store_true", help="Single poll then exit")
    p.add_argument("--massive-limit", type=int, default=250)
    p.add_argument(
        "--massive-workers",
        type=int,
        default=4,
        help="Concurrent Massive chain refresh workers",
    )
    p.add_argument(
        "--massive-cache-ttl",
        type=float,
        default=15.0,
        help="In-memory TTL in seconds for hot Massive chains",
    )
    p.add_argument(
        "--massive-hot-cache-symbols",
        default="SPY,QQQ",
        help="Comma-separated symbols eligible for RAM chain caching",
    )
    p.add_argument(
        "--paper-close",
        action="store_true",
        help="Removed path — exits with error; use --paper-close-dry-run (options are never sent to IBKR)",
    )
    p.add_argument(
        "--paper-close-dry-run",
        action="store_true",
        help="Print option close intent only (log); RLM does not broker options through IBKR",
    )
    p.add_argument(
        "--trade-log",
        type=Path,
        default=Path("data/processed/trade_log.csv"),
        help="CSV file to append per-poll P&L rows (default: data/processed/trade_log.csv).",
    )
    p.add_argument(
        "--no-trade-log",
        action="store_true",
        help="Disable trade log CSV writing.",
    )
    p.add_argument(
        "--force-close-dte",
        type=float,
        default=0.0,
        help=(
            "Force-close positions when DTE falls below this threshold (fractional days). "
            "0.0 = disabled. Recommended: 0.1 (~2.4 h) for 0DTE positions."
        ),
    )
    p.add_argument(
        "--soft-time-stop-dte",
        type=float,
        default=0.0,
        help="If >0, at/under this DTE close low-conviction positions (see --min-profit-pct-for-soft-hold).",
    )
    p.add_argument(
        "--min-profit-pct-for-soft-hold",
        type=float,
        default=20.0,
        help="Minimum unrealized PnL%% needed to keep a position in the soft time-stop zone.",
    )
    p.add_argument(
        "--max-loss-pct",
        type=float,
        default=-70.0,
        help="Max-loss kill-switch in percent; <= this unrealized PnL%% forces max_loss_stop.",
    )
    p.add_argument(
        "--rth-only-poll",
        action="store_true",
        help="Outside NYSE RTH, skip Massive polling and sleep for --interval (saves API quota)",
    )
    args = p.parse_args()

    if args.paper_close and not args.paper_close_dry_run:
        from rlm.execution.options_ibkr_policy import exit_ibkr_option_combo_blocked

        exit_ibkr_option_combo_blocked("monitor --paper-close (IBKR is equities-only; use --paper-close-dry-run)")

    def _resolve_data_path(raw: Path) -> Path:
        p = raw.expanduser()
        return p if p.is_absolute() else (DATA_ROOT / p)

    plans_path = _resolve_data_path(args.plans)
    state_path = _resolve_data_path(args.state)
    trade_log_path: Path | None = None
    if not args.no_trade_log:
        trade_log_path = _resolve_data_path(args.trade_log)

    if not plans_path.is_file():
        print(f"Missing plans file: {plans_path}", file=sys.stderr)
        return 1

    try:
        client = MassiveClient()
    except ValueError as e:
        print(f"Massive: {e}", file=sys.stderr)
        return 1

    def run_cycle() -> float:
        """Execute one monitor cycle.  Returns the recommended sleep interval (seconds)."""
        if args.rth_only_poll and not is_rth_now():
            print(
                f"[monitor] --rth-only-poll: outside NYSE RTH ({session_label()}), skipping Massive cycle",
                flush=True,
            )
            return max(5.0, float(args.interval))

        payload = _load_json(plans_path)
        ranked = list(payload.get("active_ranked") or [])
        if ranked:
            active = ranked
        else:
            active = [r for r in payload.get("results", []) if r.get("status") == "active"]
        seen: set[str] = set()
        uniq: list[dict] = []
        for r in active:
            pid = str(r.get("plan_id") or r.get("symbol"))
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append(r)

        state = _load_state(state_path)
        snap_path = state_path.with_name("trade_plan_snapshots.json")
        plan_snapshots: dict[str, dict] = {}
        if snap_path.is_file():
            try:
                loaded = json.loads(snap_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    plan_snapshots = {str(k): v for k, v in loaded.items() if isinstance(v, dict)}
            except (OSError, json.JSONDecodeError):
                plan_snapshots = {}

        # Re-evaluate open trade_log rows that fell out of universe JSON (use snapshots).
        if trade_log_path is not None:
            for oid in sorted(_open_trade_log_plan_ids(trade_log_path)):
                if oid in seen:
                    continue
                sp = plan_snapshots.get(oid)
                if not isinstance(sp, dict):
                    continue
                if not (sp.get("matched_legs") or []) and plan_combo_spec(sp) is None:
                    continue
                sym_g = str(sp.get("symbol") or "").strip().upper()
                if not sym_g:
                    continue
                ghost = {
                    "plan_id": sp.get("plan_id") or oid,
                    "symbol": sym_g,
                    "strategy": sp.get("strategy"),
                    "decision": sp.get("decision"),
                    "matched_legs": list(sp.get("matched_legs") or []),
                    "combo_spec": sp.get("combo_spec"),
                    "candidate": sp.get("candidate"),
                    "regime_key": sp.get("regime_key"),
                    "thresholds": sp.get("thresholds") or {},
                    "entry_debit_dollars": sp.get("entry_debit_dollars"),
                    "entry_mid_mark_dollars": sp.get("entry_mid_mark_dollars"),
                    "status": "open",
                }
                seen.add(oid)
                uniq.append(ghost)
                print(f"[monitor] ghost plan from snapshot: {oid} ({sym_g})", flush=True)

        by_sym: dict[str, list[dict]] = {}
        for pl in uniq:
            sym = str(pl.get("symbol", "")).upper()
            if sym:
                by_sym.setdefault(sym, []).append(pl)

        min_dte_seen = float("inf")
        hot_cache_symbols = _parse_symbols(args.massive_hot_cache_symbols)
        snapshot_params_by_symbol = {
            sym: _build_incremental_snapshot_params(plist, massive_limit=args.massive_limit)
            for sym, plist in by_sym.items()
        }
        batch = massive_option_chains_from_client(
            client,
            list(by_sym),
            per_symbol_params=snapshot_params_by_symbol,
            max_workers=max(1, int(args.massive_workers)),
            cache_ttl_s=max(0.0, float(args.massive_cache_ttl)),
            hot_cache_symbols=hot_cache_symbols,
        )

        for sym, plist in by_sym.items():
            if sym in batch.errors:
                print(f"[{sym}] Massive error: {batch.errors[sym]}", file=sys.stderr)
                continue
            chain = batch.chains.get(sym)
            if chain is None:
                print(f"[{sym}] Massive error: missing chain response", file=sys.stderr)
                continue
            for pl in plist:
                _evaluate_plan(
                    pl,
                    chain=chain,
                    state=state,
                    paper_close=bool(args.paper_close or args.paper_close_dry_run),
                    paper_close_dry_run=bool(args.paper_close_dry_run),
                    force_close_dte=args.force_close_dte,
                    soft_time_stop_dte=args.soft_time_stop_dte,
                    min_profit_pct_for_soft_hold=args.min_profit_pct_for_soft_hold,
                    max_loss_pct=args.max_loss_pct,
                    trade_log_path=trade_log_path,
                    plan_snapshots=plan_snapshots,
                )
                d = dte_from_plan(pl)
                if d == d and d >= 0:  # not NaN, not expired
                    min_dte_seen = min(min_dte_seen, d)

        _save_state(state_path, state)
        try:
            snap_path.write_text(json.dumps(plan_snapshots, indent=2, default=str), encoding="utf-8")
        except OSError as e:
            print(f"[monitor] could not write {snap_path}: {e}", file=sys.stderr)

        # Adaptive interval: short-DTE positions need faster polling
        base = max(5.0, args.interval)
        if min_dte_seen < 1.0:
            # 0DTE / intraday: poll at 1/4 of the base interval, min 15s
            return max(15.0, base / 4.0)
        return base

    next_sleep = run_cycle()
    while not args.once:
        time.sleep(next_sleep)
        next_sleep = run_cycle()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
