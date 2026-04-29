"""
Telegram push logic driven by RLM on-disk state (no changes to the trading stack).

* Universe: new **active** ``plan_id`` in ``universe_trade_plans.json`` (and optional ``session_brief.json`` for /brief).
* Options monitor: ``trade_log.csv`` (open / take-profit / exit signals).
* Equities: ``equity_positions_state.json``.
* Balances: optional ``fetch_ibkr_account_snapshot`` (requires IB Gateway + ``ibapi``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rlm.execution.exit_signals import EXIT_SIGNALS


def default_paths(root: Path) -> dict[str, Path]:
    return {
        "plans": root / "data" / "processed" / "universe_trade_plans.json",
        "trade_log": root / "data" / "processed" / "trade_log.csv",
        "equity_trade_log": root / "data" / "processed" / "equity_trade_log.csv",
        "equity_state": root / "data" / "processed" / "equity_positions_state.json",
        "state": root / "data" / "processed" / "telegram_notify_state.json",
        "session_brief": root / "data" / "processed" / "session_brief.json",
        "challenge_state": root / "data" / "challenge" / "state.json",
        "challenge_trade_log": root / "data" / "challenge" / "trade_log.csv",
    }


def _exit_reason_human(sig: str) -> str:
    return {
        "take_profit": "take profit (mark at/above target)",
        "hard_stop": "hard stop",
        "trailing_stop": "trailing stop",
        "expiry_force_close": "DTE / expiry safety close",
        "time_stop": "time-based stop (low conviction near expiry)",
        "max_loss_stop": "max-loss kill switch",
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


def build_universe_and_positions(root: Path, *, max_active: int = 12, max_positions: int = 20) -> str:
    """Universe summary plus open option rows (trade_log) and open equity rows (state json)."""
    lines: list[str] = [
        "=== Universe ===",
        build_universe_report(root, max_active=max_active),
        "",
        "=== Options (trade_log, open) ===",
    ]
    p = default_paths(root)
    latest = _latest_rows_per_plan_csv(p["trade_log"])
    opts: list[tuple[str, dict[str, str]]] = []
    for pid, row in latest.items():
        if (row.get("closed") or "0").strip() != "1":
            opts.append((pid, row))
    if not opts:
        lines.append("  (none — no rows with closed=0)")
    else:
        for pid, row in opts[:max_positions]:
            raw_pnl = row.get("unrealized_pnl_pct", "")
            pnl_val: float | None = None
            try:
                pnl_val = float(raw_pnl)
                pnl_fmt = f"{pnl_val:+.1f}%"
            except (TypeError, ValueError):
                pnl_fmt = str(raw_pnl)
            dte_val: float | None = None
            try:
                dte_val = float(row.get("dte") or "")
            except (TypeError, ValueError):
                dte_val = None
            warn: list[str] = []
            if pnl_val is not None and pnl_val <= -70.0:
                warn.append("⚠ MAX_LOSS_BREACH")
            if dte_val is not None and dte_val <= 21.0 and (pnl_val is None or pnl_val < 20.0):
                warn.append("⚠ TIME_STOP_ZONE")
            if dte_val is not None and dte_val <= 14.0:
                warn.append("⚠ FORCE_CLOSE_ZONE")
            warn_suffix = f"  {' '.join(warn)}" if warn else ""
            lines.append(
                f"  • {row.get('symbol', '?')}  plan={pid}  mark={row.get('current_mark', '')}  "
                f"PnL={pnl_fmt}  signal={row.get('signal', '')}  dte={row.get('dte', '')}{warn_suffix}"
            )
        if len(opts) > max_positions:
            lines.append(f"  … {len(opts) - max_positions} more")
    eq = _read_equity_state(p["equity_state"])
    eq_open = [(str(k), v or {}) for k, v in eq.items() if str((v or {}).get("status") or "") == "open"]
    lines.extend(["", "=== Equity (state file, open) ==="])
    if not eq_open:
        lines.append("  (none open)")
    else:
        for pid, d in eq_open[:max_positions]:
            lines.append(
                f"  • {d.get('symbol', '?')}  side={d.get('side', '?')}  qty={d.get('quantity', '')}  plan_id={pid}"
            )
        if len(eq_open) > max_positions:
            lines.append(f"  … {len(eq_open) - max_positions} more")
    return "\n".join(lines)


def _iter_active_plan_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows treated as the active universe set (matches monitor payload selection)."""
    ranked = data.get("active_ranked") or []
    if ranked:
        return [r for r in ranked if isinstance(r, dict)]
    return [r for r in (data.get("results") or []) if isinstance(r, dict) and r.get("status") == "active"]


def _active_plan_ids_from_plans(data: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for r in _iter_active_plan_rows(data):
        pid = str(r.get("plan_id") or r.get("symbol") or "")
        if pid:
            out.add(pid)
    return out


def _symbol_by_plan_id(data: dict[str, Any]) -> dict[str, str]:
    m: dict[str, str] = {}
    for r in _iter_active_plan_rows(data):
        pid = str(r.get("plan_id") or r.get("symbol") or "")
        if not pid:
            continue
        m[pid] = str(r.get("symbol") or "?")
    return m


def build_session_brief_text(root: Path, *, max_active: int = 12) -> str:
    """Summary of ``session_brief.json`` (systemd pre/post-close pipeline output)."""
    p = default_paths(root)["session_brief"]
    if not p.is_file():
        return f"No session brief file: {p.name} (run scripts/run_session_brief.py or timers)"
    data = _read_plans(p)
    if not data:
        return f"Empty or unreadable: {p.name}"
    gen = str(data.get("generated_at_utc", "?"))
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat() if p.is_file() else "?"
    head = f"=== session_brief.json ===\ngenerated_at: {gen}\nfile mtime (UTC): {mtime}\n\n"
    return head + build_universe_report_from_data(data, max_active=max_active)


def build_universe_report_from_data(data: dict[str, Any], *, max_active: int = 12) -> str:
    """Like :func:`build_universe_report` but from an already-loaded plans payload."""
    if not data:
        return "No data"
    gen = str(data.get("generated_at_utc", "?"))
    actives: list[dict[str, Any]] = list(_iter_active_plan_rows(data))
    actives.sort(key=lambda x: float(x.get("rank_score") or 0.0), reverse=True)
    lines = [
        f"Universe report (top {min(max_active, len(actives))} of {len(actives)} active)\n" f"generated_at: {gen}\n"
    ]
    for r in actives[:max_active]:
        sym = r.get("symbol", "?")
        st = r.get("strategy", "?")
        rs = r.get("rank_score")
        conf = r.get("regime_confidence", r.get("confidence"))
        pid = r.get("plan_id", "")
        try:
            rs_fmt = f"{float(rs):.4f}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            rs_fmt = str(rs or "?")
        try:
            conf_fmt = f"{float(conf) * 100:.1f}%"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            conf_fmt = "?"
        lines.append(f"  • {sym}  {st}  score={rs_fmt}  conf={conf_fmt}  id={pid}")
    if not actives:
        lines.append("  (no active rows)")
    return "\n".join(lines)


def build_universe_report(root: Path, *, max_active: int = 12) -> str:
    p = default_paths(root)["plans"]
    data = _read_plans(p)
    if not data:
        return f"No plans file or empty: {p.name}"
    return build_universe_report_from_data(data, max_active=max_active)


def _load_all_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read every row from a CSV trade log (not just latest per plan)."""
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _pnl_aggregates_from_log(
    rows: list[dict[str, str]],
) -> tuple[float, float, float, float]:
    """Compute (daily, weekly, all_time, open_mtm) from a trade log CSV.

    Closed trades: last row per plan_id with closed=1 → realized PnL.
    Open positions: last row per plan_id with closed!=1 → unrealized MTM.
    """
    if not rows:
        return 0.0, 0.0, 0.0, 0.0

    now = datetime.now().astimezone()
    today = now.date()
    iso_now = today.isocalendar()

    # Two passes: (1) collect all closed rows for realized PnL (one per
    # plan_id — the *last* closed snapshot), (2) latest row per plan_id
    # that is still open for unrealized MTM.
    closed_by_pid: dict[str, dict[str, str]] = {}
    latest_by_pid: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = str(row.get("plan_id") or "")
        if not pid:
            continue
        latest_by_pid[pid] = row
        if (row.get("closed") or "0").strip() == "1":
            closed_by_pid[pid] = row

    daily = 0.0
    weekly = 0.0
    all_time = 0.0
    open_mtm = 0.0

    for pid, row in closed_by_pid.items():
        try:
            pnl = float(row.get("unrealized_pnl") or 0)
        except (ValueError, TypeError):
            pnl = 0.0
        all_time += pnl
        ts_raw = row.get("timestamp_utc", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            local_date = ts.astimezone().date()
            if local_date == today:
                daily += pnl
            iso_ts = local_date.isocalendar()
            if iso_ts.year == iso_now.year and iso_ts.week == iso_now.week:
                weekly += pnl
        except (ValueError, TypeError):
            pass

    for pid, row in latest_by_pid.items():
        if (row.get("closed") or "0").strip() == "1":
            continue
        try:
            open_mtm += float(row.get("unrealized_pnl") or 0)
        except (ValueError, TypeError):
            pass

    return daily, weekly, all_time, open_mtm


def _fmt_pnl(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.2f}"


def _challenge_pnl_section(root: Path) -> str:
    """Build the PDT Challenge section of the PnL report."""
    p = default_paths(root)
    state_path = p["challenge_state"]
    if not state_path.is_file():
        return "--- PDT CHALLENGE ($1K -> $25K) ---\n  (no challenge state — run `rlm challenge --reset`)"

    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "--- PDT CHALLENGE ($1K -> $25K) ---\n  (unreadable state file)"

    balance = float(raw.get("balance", 0))
    seed = float(raw.get("seed", 1000))
    target = float(raw.get("target", 25000))
    span = target - seed
    progress = min(100.0, max(0.0, (balance - seed) / span * 100.0)) if span > 0 else 100.0

    from rlm.challenge.config import MILESTONES

    milestone_label = "—"
    for m in MILESTONES:
        if balance < m.target:
            milestone_label = m.label
            break
    else:
        milestone_label = MILESTONES[-1].label if MILESTONES else "—"

    now = datetime.now().astimezone()
    today = now.date()
    iso_now = today.isocalendar()

    daily = 0.0
    weekly = 0.0
    all_time = balance - seed

    trade_history = raw.get("trade_history") or []
    for t in trade_history:
        try:
            pnl = float(t.get("pnl", 0))
        except (ValueError, TypeError):
            continue
        exit_date_raw = str(t.get("exit_date", ""))
        try:
            exit_dt = datetime.fromisoformat(exit_date_raw.replace("Z", "+00:00"))
            if exit_dt.tzinfo is None:
                exit_dt = exit_dt.replace(tzinfo=timezone.utc)
            local_date = exit_dt.astimezone().date()
            if local_date == today:
                daily += pnl
            iso_d = local_date.isocalendar()
            if iso_d.year == iso_now.year and iso_d.week == iso_now.week:
                weekly += pnl
        except (ValueError, TypeError):
            pass

    open_mtm = 0.0
    for pos in raw.get("open_positions") or []:
        try:
            open_mtm += float(pos.get("unrealised_pnl", 0))
        except (ValueError, TypeError):
            pass

    lines = [
        "--- PDT CHALLENGE ($1K -> $25K) ---",
        f"Balance:   ${balance:,.2f}",
        f"Seed:      ${seed:,.2f}",
        f"Progress:  {progress:.1f}% ({milestone_label})",
        f"Today:     {_fmt_pnl(daily)}",
        f"This week: {_fmt_pnl(weekly)}",
        f"All-time:  {_fmt_pnl(all_time)}",
    ]
    if open_mtm:
        lines.append(f"Open MTM:  {_fmt_pnl(open_mtm)}")
    n_trades = len(trade_history)
    if n_trades:
        wins = 0
        for t in trade_history:
            try:
                if float(t.get("pnl", 0)) > 0:
                    wins += 1
            except (ValueError, TypeError):
                pass
        wr = wins / n_trades * 100
        lines.append(f"Trades: {n_trades}  W/L: {wins}/{n_trades - wins}  WR: {wr:.0f}%")
    return "\n".join(lines)


def _ibkr_balance_section() -> str:
    """Compact IBKR account balances for the PnL report."""
    try:
        from rlm.data.ibkr_snapshot import fetch_ibkr_account_snapshot
    except ImportError:
        return "--- IBKR ACCOUNT ---\n  (ibapi not installed)"
    try:
        snap = fetch_ibkr_account_snapshot(timeout_sec=15.0)
    except Exception:  # noqa: BLE001
        return "--- IBKR ACCOUNT ---\n  (Gateway unavailable)"

    def _tag(t: str) -> str:
        for row in snap.account_summary:
            if str(row.tag) == t and row.value:
                try:
                    return f"${float(row.value):,.2f}"
                except (ValueError, TypeError):
                    return f"{row.value} {row.currency or ''}".strip()
        return "—"

    return "\n".join(
        [
            "--- IBKR ACCOUNT ---",
            f"Net liq:    {_tag('NetLiquidation')}",
            f"Cash:       {_tag('TotalCashValue')}",
            f"Buying pwr: {_tag('BuyingPower')}",
            f"Unreal PnL: {_tag('UnrealizedPnL')}",
            f"Real PnL:   {_tag('RealizedPnL')}",
        ]
    )


def build_pnl_text(root: Path) -> str:
    """Full P&L report across all three systems: equities, options, PDT challenge + IBKR balances."""
    p = default_paths(root)
    sections: list[str] = ["=== P&L REPORT ==="]

    eq_rows = _load_all_csv_rows(p["equity_trade_log"])
    eq_d, eq_w, eq_a, eq_mtm = _pnl_aggregates_from_log(eq_rows)
    sections.append(
        "\n".join(
            [
                "",
                "--- EQUITIES ---",
                f"Today:     {_fmt_pnl(eq_d)}",
                f"This week: {_fmt_pnl(eq_w)}",
                f"All-time:  {_fmt_pnl(eq_a)}",
                f"Open MTM:  {_fmt_pnl(eq_mtm)}",
            ]
        )
    )

    opt_rows = _load_all_csv_rows(p["trade_log"])
    opt_d, opt_w, opt_a, opt_mtm = _pnl_aggregates_from_log(opt_rows)
    sections.append(
        "\n".join(
            [
                "",
                "--- OPTIONS (Large Acct) ---",
                f"Today:     {_fmt_pnl(opt_d)}",
                f"This week: {_fmt_pnl(opt_w)}",
                f"All-time:  {_fmt_pnl(opt_a)}",
                f"Open MTM:  {_fmt_pnl(opt_mtm)}",
            ]
        )
    )

    sections.append("")
    sections.append(_challenge_pnl_section(root))

    sections.append("")
    sections.append(_ibkr_balance_section())

    return "\n".join(sections)


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
    opt: list[IbkrPositionRow] = [
        x for x in snap.positions if str(x.sec_type).upper() in {"OPT", "BAG", "BOND"} and abs(x.position) > 0
    ]

    lines = [
        f"IBKR @ {snap.host}:{snap.port} (client {snap.client_id})",
        f"Net liq: {nlv}  |  Cash: {cash}",
        f"Buying power: {bp}  |  Unrealized PnL: {u_pnl}",
        f"Equity positions (STK): {len(stk)}  |  Option legs / non-stock: {len(opt)}",
    ]
    for pr in stk[:8]:
        lines.append(f"  STK: {pr.symbol}  qty={pr.position}  avg={pr.avg_cost:.2f}  ccy={pr.currency}")
    if len(stk) > 8:
        lines.append(f"  … {len(stk) - 8} more stock rows")
    for pr in opt[:8]:
        lines.append(f"  OPT: {pr.symbol}  qty={pr.position}  avg={pr.avg_cost:.2f}  ccy={pr.currency}")
    if len(opt) > 8:
        lines.append(f"  … {len(opt) - 8} more option rows")
    return "\n".join(lines)


@dataclass
class _St:
    notify_seeded: bool = False
    """Plan IDs with closed=0 in trade_log we have already announced as an open position."""
    announced_trade_open: set[str] = field(default_factory=set)
    last_opt_signal: dict[str, str] = field(default_factory=dict)
    announced_tp: set[str] = field(default_factory=set)
    announced_exit: set[str] = field(default_factory=set)
    last_equity_open: set[str] = field(default_factory=set)
    announced_equity_close: set[str] = field(default_factory=set)
    last_universe_active_ids: set[str] = field(default_factory=set)

    @staticmethod
    def from_json(d: dict[str, Any]) -> _St:
        s = _St()
        s.notify_seeded = bool(d.get("notify_seeded", d.get("seeded", False)))
        raw_ato = d.get("announced_trade_open")
        if raw_ato is not None:
            s.announced_trade_open = set(str(x) for x in raw_ato)
        s.last_opt_signal = {str(k): str(v) for k, v in (d.get("last_opt_signal") or {}).items()}
        s.announced_tp = set(str(x) for x in (d.get("announced_tp") or []))
        s.announced_exit = set(str(x) for x in (d.get("announced_exit") or []))
        s.last_equity_open = set(str(x) for x in (d.get("last_equity_open") or []))
        s.announced_equity_close = set(str(x) for x in (d.get("announced_equity_close") or []))
        raw_u = d.get("last_universe_active_ids")
        if raw_u is not None:
            s.last_universe_active_ids = set(str(x) for x in raw_u)
        return s

    def to_json(self) -> dict[str, Any]:
        return {
            "notify_seeded": self.notify_seeded,
            "announced_trade_open": sorted(self.announced_trade_open),
            "last_opt_signal": self.last_opt_signal,
            "announced_tp": sorted(self.announced_tp),
            "announced_exit": sorted(self.announced_exit),
            "last_equity_open": sorted(self.last_equity_open),
            "announced_equity_close": sorted(self.announced_equity_close),
            "last_universe_active_ids": sorted(self.last_universe_active_ids),
        }


def notification_cycle(root: Path, state_blob: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """
    Return (outbound messages, new state fields) for merging into the full on-disk state dict.
    """
    p = default_paths(root)
    st = _St.from_json(state_blob)
    out: list[str] = []

    latest = _latest_rows_per_plan_csv(p["trade_log"])
    eq = _read_equity_state(p["equity_state"])
    now_open: set[str] = set()
    for pid, d in eq.items():
        pkey = str(pid)
        if str((d or {}).get("status") or "") == "open":
            now_open.add(pkey)

    if not st.notify_seeded:
        for pid, row in latest.items():
            sig = (row.get("signal") or "hold").strip()
            st.last_opt_signal[pid] = sig
            cl = (row.get("closed") or "0").strip() == "1"
            if cl and sig in EXIT_SIGNALS:
                st.announced_exit.add(pid)
            if sig == "take_profit":
                st.announced_tp.add(pid)
        st.announced_trade_open = {pid for pid, row in latest.items() if (row.get("closed") or "0").strip() != "1"}
        st.last_equity_open = set(now_open)
        plans_data0 = _read_plans(p["plans"])
        st.last_universe_active_ids = _active_plan_ids_from_plans(plans_data0)
        st.notify_seeded = True
        merged = {**state_blob, **st.to_json()}
        return [], merged

    # Upgrade older state files that used known_option_plans but not announced_trade_open
    if st.notify_seeded and "announced_trade_open" not in state_blob:
        st.announced_trade_open = {pid for pid, row in latest.items() if (row.get("closed") or "0").strip() != "1"}
        merged = {**state_blob, **st.to_json()}
        return [], merged

    if st.notify_seeded and "last_universe_active_ids" not in state_blob:
        plans_data_u = _read_plans(p["plans"])
        st.last_universe_active_ids = _active_plan_ids_from_plans(plans_data_u)
        merged = {**state_blob, **st.to_json()}
        return [], merged

    for pid, row in latest.items():
        sig = (row.get("signal") or "hold").strip()
        mark = row.get("current_mark", "")
        sym = row.get("symbol", "")
        closed = (row.get("closed") or "0").strip() == "1"
        prev = st.last_opt_signal.get(pid, "")

        if not closed and pid not in st.announced_trade_open:
            st.announced_trade_open.add(pid)
            ed = row.get("entry_debit", row.get("entry_mid", ""))
            out.append(f"Alert: New position — {sym}  plan={pid}  mark={mark}  entry~{ed}  signal={sig}")

        if sig == "take_profit" and prev != "take_profit" and pid not in st.announced_tp:
            st.announced_tp.add(pid)
            out.append(f"Alert: Position now above profit target — {sym}  plan={pid}  mark={mark}")
        if closed and sig in EXIT_SIGNALS and pid not in st.announced_exit:
            st.announced_exit.add(pid)
            st.announced_trade_open.discard(pid)
            out.append(f"Alert: Exited position — {sym}  plan={pid}  reason={_exit_reason_human(sig)}  mark={mark}")
        st.last_opt_signal[pid] = sig

    for gone in set(st.announced_tp) - set(latest.keys()):
        st.announced_tp.discard(gone)
    for gone in set(st.announced_exit) - set(latest.keys()):
        st.announced_exit.discard(gone)
    for pid in list(st.announced_trade_open):
        row = latest.get(pid)
        if not row or (row.get("closed") or "0").strip() == "1":
            st.announced_trade_open.discard(pid)

    plans_data = _read_plans(p["plans"])
    cur_u = _active_plan_ids_from_plans(plans_data)
    _symbol_by_plan_id(plans_data)
    # Disabling universe idea alerts to reduce chatter per user request
    # for pid in sorted(cur_u - st.last_universe_active_ids):
    #     out.append(
    #         f"Alert: New active universe idea — {sym_by_u.get(pid, '?')}  plan={pid} "
    #         "(universe_trade_plans.json)"
    #     )
    st.last_universe_active_ids = cur_u

    prev_eq = st.last_equity_open
    for pid, d in eq.items():
        pkey = str(pid)
        pdat = d or {}
        st_eq = str(pdat.get("status") or "")
        if st_eq == "open":
            if pkey not in prev_eq and pkey not in st.announced_equity_close:
                out.append(
                    f"Alert: New position — {pdat.get('symbol', '?')}  "
                    f"side={pdat.get('side', '?')}  qty={pdat.get('quantity', '')}  "
                    f"plan_id={pkey}  (equity)"
                )
        elif st_eq == "closed" and pkey in prev_eq and pkey not in st.announced_equity_close:
            st.announced_equity_close.add(pkey)
            ex = pdat.get("exit_reason") or pdat.get("note") or "—"
            out.append(f"Alert: Exited position — {pdat.get('symbol', '?')}  plan_id={pkey}  reason={ex}  (equity)")

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
