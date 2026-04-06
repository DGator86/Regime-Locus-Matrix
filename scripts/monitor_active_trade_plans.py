#!/usr/bin/env python3
"""
Poll **Massive** option snapshots for open trade plans and evaluate **mid-mark** vs exit rules:

- **Take profit** when ``V >= v_take_profit``
- **Hard stop** when ``V <= v_hard_stop``
- **Trailing stop** after ``V`` reaches ``v_trail_activate``: maintain ``peak``; exit if
  ``V < peak * (1 - trail_retrace_frac)``

Input: JSON from ``scripts/run_universe_options_pipeline.py`` (``--out``).

State file (repo root relative): ``data/processed/trade_monitor_state.json`` tracks ``peak_v`` and
``trail_on`` per ``plan_id``.

With ``--paper-close`` (paper **7497** / **4002** only), submits a **market** combo in the opposite
direction of ``ibkr_combo_spec`` once per plan when an exit **ACTION** fires.

Examples::

    python scripts/monitor_active_trade_plans.py --plans data/processed/universe_trade_plans.json --once
    python scripts/monitor_active_trade_plans.py --plans data/processed/universe_trade_plans.json --interval 120
    python scripts/monitor_active_trade_plans.py --plans data/processed/universe_trade_plans.json --paper-close --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.data.massive import MassiveClient
from rlm.data.massive_option_chain import massive_option_chain_from_client
from rlm.execution.ibkr_combo_orders import (
    assert_paper_trading_port,
    legs_from_ibkr_combo_spec,
    load_ibkr_order_socket_config,
    place_options_combo_market_order,
    reverse_legs_for_close,
)
from rlm.execution.risk_targets import trailing_stop_from_peak
from rlm.roee.chain_match import estimate_mark_value_from_matched_legs


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


def _contract_key(leg: dict) -> str:
    return str(leg.get("symbol") or leg.get("contract_symbol") or "")


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


def _evaluate_plan(
    plan: dict,
    *,
    chain: pd.DataFrame,
    state: dict,
    paper_close: bool,
    paper_close_dry_run: bool,
) -> None:
    pid = str(plan.get("plan_id") or plan.get("symbol") or "unknown")
    thr = plan.get("thresholds") or {}
    mlegs = plan.get("matched_legs") or []
    sym = str(plan.get("symbol", ""))

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

    msg = f"[{sym}] {pid} V={v:.2f}  V0={plan.get('entry_mid_mark_dollars')}  debit={plan.get('entry_debit_dollars')}"

    signal = "hold"
    if v >= v_tp:
        signal = "take_profit"
    elif v <= v_sl:
        signal = "hard_stop"
    elif st.get("trail_on"):
        peak = float(st["peak_v"])
        tstop = trailing_stop_from_peak(peak, tr_r)
        if v < tstop:
            signal = "trailing_stop"

    if signal == "take_profit":
        print(f"{msg}  ACTION: TAKE_PROFIT (V >= {v_tp:.2f})")
    elif signal == "hard_stop":
        print(f"{msg}  ACTION: HARD_STOP (V <= {v_sl:.2f})")
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

    exit_signals = frozenset({"take_profit", "hard_stop", "trailing_stop"})
    if (
        paper_close
        and signal in exit_signals
        and not st.get("paper_close_sent")
        and isinstance(plan.get("ibkr_combo_spec"), dict)
    ):
        spec = plan["ibkr_combo_spec"]
        qty = int(spec.get("quantity", 1))
        try:
            open_legs = legs_from_ibkr_combo_spec(spec)
            close_legs = reverse_legs_for_close(open_legs)
        except Exception as e:
            print(f"[{sym}] {pid} PAPER-CLOSE skip (bad spec): {e}", file=sys.stderr)
            return

        oa = str(spec.get("combo_order_action", "BUY")).upper()
        close_parent = "BUY" if oa == "SELL" else "SELL"

        if paper_close_dry_run:
            print(f"[{sym}] {pid} PAPER-CLOSE DRY-RUN MKT {close_parent} qty={qty} legs={len(close_legs)}")
            st["paper_close_sent"] = True
            return

        try:
            oid, trail = place_options_combo_market_order(
                close_legs,
                quantity=qty,
                transmit=True,
                acknowledge_live=False,
                combo_order_action=close_parent,  # type: ignore[arg-type]
            )
            print(f"[{sym}] {pid} PAPER-CLOSE orderId={oid} trail={trail}")
            st["paper_close_sent"] = True
        except Exception as e:
            print(f"[{sym}] {pid} PAPER-CLOSE FAILED: {e}", file=sys.stderr)

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plans", type=Path, required=True, help="universe_trade_plans.json")
    p.add_argument(
        "--state",
        type=Path,
        default=Path("data/processed/trade_monitor_state.json"),
        help="Persistent peak/trail state",
    )
    p.add_argument("--interval", type=float, default=120.0, help="Seconds between polls (if not --once)")
    p.add_argument("--once", action="store_true", help="Single poll then exit")
    p.add_argument("--massive-limit", type=int, default=250)
    p.add_argument(
        "--paper-close",
        action="store_true",
        help="On exit ACTION, transmit IBKR **MKT** closing combo (paper port only)",
    )
    p.add_argument(
        "--paper-close-dry-run",
        action="store_true",
        help="Print close intent only (no IBKR); does not require paper port",
    )
    args = p.parse_args()

    plans_path = ROOT / args.plans if not args.plans.is_absolute() else args.plans
    state_path = ROOT / args.state if not args.state.is_absolute() else args.state

    if not plans_path.is_file():
        print(f"Missing plans file: {plans_path}", file=sys.stderr)
        return 1

    try:
        client = MassiveClient()
    except ValueError as e:
        print(f"Massive: {e}", file=sys.stderr)
        return 1

    def run_cycle() -> None:
        if args.paper_close and not args.paper_close_dry_run:
            _, port, _ = load_ibkr_order_socket_config()
            assert_paper_trading_port(port)

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

        by_sym: dict[str, list[dict]] = {}
        for pl in uniq:
            sym = str(pl.get("symbol", "")).upper()
            if sym:
                by_sym.setdefault(sym, []).append(pl)

        for sym, plist in by_sym.items():
            try:
                chain = massive_option_chain_from_client(client, sym, limit=args.massive_limit)
            except Exception as e:
                print(f"[{sym}] Massive error: {e}", file=sys.stderr)
                continue
            for pl in plist:
                _evaluate_plan(
                    pl,
                    chain=chain,
                    state=state,
                    paper_close=bool(args.paper_close or args.paper_close_dry_run),
                    paper_close_dry_run=bool(args.paper_close_dry_run),
                )

        _save_state(state_path, state)

    run_cycle()
    while not args.once:
        time.sleep(max(5.0, args.interval))
        run_cycle()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
