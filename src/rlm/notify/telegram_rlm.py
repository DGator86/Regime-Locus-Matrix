"""
Telegram push logic driven by RLM on-disk state (no changes to the trading stack).

* Options: ``trade_log.csv`` (monitor) + ``universe_trade_plans.json``
* Equities: ``equity_positions_state.json``
* Balances: optional ``fetch_ibkr_account_snapshot`` (requires IB Gateway + ``ibapi``)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def default_paths(root: Path) -> dict[str, Path]:
    return {
        "plans": root / "data" / "processed" / "universe_trade_plans.json",
        "trade_log": root / "data" / "processed" / "trade_log.csv",
        "equity_state": root / "data" / "processed" / "equity_positions_state.json",
        "state": root / "data" / "processed" / "telegram_notify_state.json",
    }


EXIT_SIGNALS = frozenset({"take_profit", "hard_stop", "trailing_stop", "expiry_force_close"})


def _exit_reason_human(sig: str) -> str:
    return {
        "take_profit": "take profit (mark at/above target)",
        "hard_stop": "hard stop",
        "trailing_stop": "trailing stop",
        "expiry_force_close": "DTE / expiry safety close",
    }.get(sig, sig)


def _read_plans(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _latest_rows_per_plan_csv(path: Path) -> dict[str, dict[str, str]]:
    """Last row for each plan_id in trade log."""
    if not path.is_file():
        return {}
    by_pid: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                pid = str(row.get("plan_id") or "")
                if pid:
                    by_pid[pid] = {k: str(v) for k, v in row.items()}
    except OSError:
        return {}
    return by_pid


def _read_equity_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_notify_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_notify_state(path: Path, d: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")


def build_status_brief(root: Path) -> str:
    p = default_paths(root)["plans"]
    data = _read_plans(p)
    if not data:
        return f"No plans file or empty: {p.name}"
    gen = str(data.get("generated_at_utc", "?"))
    results = data.get("results") or []
    n_active = sum(1 for r in results if r.get("status") == "active")
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat() if p.is_file() else "?"
    return f"generated_at: {gen}\nfile mtime (UTC): {mtime}\nactive: {n_active}"


def build_universe_report(root: Path, *, max_active: int = 12) -> str:
    p = default_paths(root)["plans"]
    data = _read_plans(p)
    if not data:
        return f"No plans file or empty: {p.name}"
    gen = str(data.get("generated_at_utc", "?"))
    actives: list[dict[str, Any]] = []
    for r in data.get("results") or []:
        if r.get("status") == "active":
            actives.append(r)
    actives.sort(key=lambda x: float(x.get("rank_score") or 0.0), reverse=True)
    lines = [
        f"Universe report (top {min(max_active, len(actives))} of {len(actives)} active)\n"
        f"generated_at: {gen}\n"
    ]
    for r in actives[:max_active]:
        sym = r.get("symbol", "?")
        st = r.get("strategy", "?")
        rs = r.get("rank_score", "")
        pid = r.get("plan_id", "")
        lines.append(f"  • {sym}  {st}  score={rs}  id={pid}")
    if not actives:
        lines.append("  (no active rows)")
    return "\n".join(lines)


def build_balances_text(root: Path) -> str:
    """IBKR one-shot snapshot; one paper account — split by STK vs OPT position rows."""
    try:
        from rlm.data.ibkr_snapshot import IbkrPositionRow, fetch_ibkr_account_snapshot
    except ImportError as e:
        return f"IBKR not available: {e}"
    try:
        snap = fetch_ibkr_account_snapshot(timeout_sec=25.0)
    except Exception as e:
        return f"Could not read IBKR balances: {e}\n(Confirm Gateway is up and .env has IBKR_HOST/PORT.)"

    def _tag(t: str) -> str:
        for row in snap.account_summary:
            if str(row.tag) == t and row.value:
                return f"{row.value} {row.currency or ''}".strip()
        return "—"

    nlv = _tag("NetLiquidation")
    cash = _tag("TotalCashValue")
    bp = _tag("BuyingPower")
    u_pnl = _tag("UnrealizedPnL")

    stk: list[IbkrPositionRow] = [x for x in snap.positions if str(x.sec_type).upper() == "STK" and abs(x.position) > 0]
    opt: list[IbkrPositionRow] = [x for x in snap.positions if str(x.sec_type).upper() in {"OPT", "BAG", "BOND"} and abs(x.position) > 0]

    lines = [
        f"IBKR @ {snap.host}:{snap.port} (client {snap.client_id})",
        f"Net liq: {nlv}  |  Cash: {cash}",
        f"Buying power: {bp}  |  Unrealized PnL: {u_pnl}",
        f"Equity positions (STK): {len(stk)}  |  Option legs / non-stock: {len(opt)}",
    ]
    for pr in stk[:8]:
        lines.append(
            f"  STK: {pr.symbol}  qty={pr.position}  avg={pr.avg_cost:.2f}  ccy={pr.currency}"
        )
    if len(stk) > 8:
        lines.append(f"  … {len(stk) - 8} more stock rows")
    for pr in opt[:8]:
        lines.append(
            f"  OPT: {pr.symbol}  qty={pr.position}  avg={pr.avg_cost:.2f}  ccy={pr.currency}"
        )
    if len(opt) > 8:
        lines.append(f"  … {len(opt) - 8} more option rows")
    return "\n".join(lines)


@dataclass
class _St:
    notify_seeded: bool = False
    known_option_plans: set[str] = field(default_factory=set)
    last_opt_signal: dict[str, str] = field(default_factory=dict)
    announced_tp: set[str] = field(default_factory=set)
    announced_exit: set[str] = field(default_factory=set)
    last_equity_open: set[str] = field(default_factory=set)
    announced_equity_close: set[str] = field(default_factory=set)

    @staticmethod
    def from_json(d: dict[str, Any]) -> _St:
        s = _St()
        s.notify_seeded = bool(d.get("notify_seeded", d.get("seeded", False)))
        s.known_option_plans = set(str(x) for x in (d.get("known_option_plans") or []))
        s.last_opt_signal = {str(k): str(v) for k, v in (d.get("last_opt_signal") or {}).items()}
        s.announced_tp = set(str(x) for x in (d.get("announced_tp") or []))
        s.announced_exit = set(str(x) for x in (d.get("announced_exit") or []))
        s.last_equity_open = set(str(x) for x in (d.get("last_equity_open") or []))
        s.announced_equity_close = set(str(x) for x in (d.get("announced_equity_close") or []))
        return s

    def to_json(self) -> dict[str, Any]:
        return {
            "notify_seeded": self.notify_seeded,
            "known_option_plans": sorted(self.known_option_plans),
            "last_opt_signal": self.last_opt_signal,
            "announced_tp": sorted(self.announced_tp),
            "announced_exit": sorted(self.announced_exit),
            "last_equity_open": sorted(self.last_equity_open),
            "announced_equity_close": sorted(self.announced_equity_close),
        }


def notification_cycle(root: Path, state_blob: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """
    Return (outbound messages, new state fields) for merging into the full on-disk state dict.
    """
    p = default_paths(root)
    st = _St.from_json(state_blob)
    out: list[str] = []

    plans_data = _read_plans(p["plans"])
    active_ids: set[str] = set()
    for r in plans_data.get("results") or []:
        if r.get("status") == "active":
            pid = str(r.get("plan_id") or r.get("symbol") or "")
            if pid:
                active_ids.add(pid)

    latest = _latest_rows_per_plan_csv(p["trade_log"])
    eq = _read_equity_state(p["equity_state"])
    now_open: set[str] = set()
    for pid, d in eq.items():
        pkey = str(pid)
        if str((d or {}).get("status") or "") == "open":
            now_open.add(pkey)

    if not st.notify_seeded:
        st.known_option_plans = set(active_ids)
        for pid, row in latest.items():
            sig = (row.get("signal") or "hold").strip()
            st.last_opt_signal[pid] = sig
            cl = (row.get("closed") or "0").strip() == "1"
            if cl and sig in EXIT_SIGNALS:
                st.announced_exit.add(pid)
            if sig == "take_profit":
                st.announced_tp.add(pid)
        st.last_equity_open = set(now_open)
        st.notify_seeded = True
        merged = {**state_blob, **st.to_json()}
        return [], merged

    for r in plans_data.get("results") or []:
        if r.get("status") != "active":
            continue
        pid = str(r.get("plan_id") or r.get("symbol") or "")
        if not pid or pid in st.known_option_plans:
            continue
        st.known_option_plans.add(pid)
        sym = r.get("symbol", "?")
        strat = r.get("strategy", "?")
        v0 = r.get("entry_debit_dollars", "")
        out.append(
            f"[Options] New active plan: {sym}  strategy={strat}  id={pid}  entry_debit~{v0}"
        )

    for pid, row in latest.items():
        sig = (row.get("signal") or "hold").strip()
        mark = row.get("current_mark", "")
        sym = row.get("symbol", "")
        closed = (row.get("closed") or "0").strip() == "1"
        prev = st.last_opt_signal.get(pid, "")

        if sig == "take_profit" and prev != "take_profit" and pid not in st.announced_tp:
            st.announced_tp.add(pid)
            out.append(
                f"[Options] {sym}  past take-profit  plan={pid}  mark={mark}  (V >= target)"
            )
        if closed and sig in EXIT_SIGNALS and pid not in st.announced_exit:
            st.announced_exit.add(pid)
            out.append(
                f"[Options] {sym}  CLOSED  plan={pid}  reason={_exit_reason_human(sig)}  mark={mark}"
            )
        st.last_opt_signal[pid] = sig

    for gone in set(st.announced_tp) - set(latest.keys()):
        st.announced_tp.discard(gone)
    for gone in set(st.announced_exit) - set(latest.keys()):
        st.announced_exit.discard(gone)

    prev_eq = st.last_equity_open
    for pid, d in eq.items():
        pkey = str(pid)
        pdat = d or {}
        st_eq = str(pdat.get("status") or "")
        if st_eq == "open":
            if pkey not in prev_eq and pkey not in st.announced_equity_close:
                out.append(
                    f"[Equity] New position: {pdat.get('symbol', '?')}  "
                    f"side={pdat.get('side', '?')}  qty={pdat.get('quantity', '')}  "
                    f"plan_id={pkey}"
                )
        elif st_eq == "closed" and pkey in prev_eq and pkey not in st.announced_equity_close:
            st.announced_equity_close.add(pkey)
            ex = pdat.get("exit_reason") or pdat.get("note") or "—"
            out.append(
                f"[Equity] CLOSED {pdat.get('symbol', '?')}  plan_id={pkey}  reason={ex}"
            )

    st.last_equity_open = now_open
    return out, {**state_blob, **st.to_json()}


def run_notification_loop(
    root: Path,
    send: Callable[[str], None],
    state_path: Path,
    interval_sec: float = 20.0,
) -> None:
    """Background loop: run forever (caller in daemon thread), send messages on changes."""
    import time

    while True:
        blob = load_notify_state(state_path)
        try:
            messages, new_blob = notification_cycle(root, blob)
            for m in messages:
                send(m)
            if new_blob != blob:
                save_notify_state(state_path, new_blob)
        except Exception as e:  # noqa: BLE001
            print(f"[rlm-telegram] notify cycle error: {e}", flush=True)
        time.sleep(max(5.0, float(interval_sec)))
