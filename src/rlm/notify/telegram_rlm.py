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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from rlm.data.occ_symbol import format_occ_compact_symbol
from rlm.execution.combo_spec import plan_combo_spec
from rlm.execution.exit_signals import EXIT_SIGNALS
from rlm.notify.ledger_books import equity_book_snapshot, options_book_snapshot, write_trading_ledgers
from rlm.notify.options_paths import options_trade_log_primary, options_trade_log_read_paths
from rlm.universe.active_plans import active_plan_ids as _active_plan_ids_from_plans_payload
from rlm.universe.active_plans import iter_active_trade_plan_rows as _iter_active_trade_plan_rows


def _resolved_trade_log_for_notify(root: Path) -> Path:
    """Prefer the first monitor log that actually has body rows (primary may be header-only)."""
    for cand in options_trade_log_read_paths(root):
        if cand.is_file() and cand.stat().st_size > 120:
            return cand
    return options_trade_log_primary(root)


def default_paths(root: Path) -> dict[str, Path]:
    return {
        "plans": root / "data" / "processed" / "universe_trade_plans.json",
        "trade_log": _resolved_trade_log_for_notify(root),
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


def _regime_human(regime_key: str) -> str:
    """Convert pipe-delimited regime key to plain English."""
    if not regime_key:
        return "Unknown"
    parts = regime_key.split("|")
    direction_map = {
        "bull": "Bullish trend",
        "bear": "Bearish trend",
        "range": "Range-bound",
        "transition": "Transitioning",
    }
    vol_map = {
        "low_vol": "low volatility",
        "high_vol": "elevated volatility",
        "transition": "volatility transitioning",
    }
    liq_map = {
        "high_liquidity": "good liquidity",
        "low_liquidity": "thin liquidity",
    }
    flow_map = {
        "supportive": "dealer flow supportive",
        "destabilizing": "dealer flow destabilizing",
    }
    maps = [direction_map, vol_map, liq_map, flow_map]
    labels = []
    for i, part in enumerate(parts[:4]):
        p = part.strip()
        label = maps[i].get(p, p) if i < len(maps) else p
        if label:
            labels.append(label)
    return " · ".join(labels) if labels else regime_key


def _strategy_entry_human(strategy_name: str) -> str:
    """Map strategy_name to a human-readable entry logic description."""
    _MAP = {
        "long_call_spread": "Long call debit spread — defined-risk bullish play",
        "bull_call_spread": "Bull call debit spread — defined-risk bullish play",
        "0dte_bull_call_spread": "0DTE bull call spread — intraday bullish sniper",
        "long_put_spread": "Long put debit spread — defined-risk bearish play",
        "bear_put_spread": "Bear put debit spread — defined-risk bearish play",
        "0dte_bear_put_spread": "0DTE bear put spread — intraday bearish sniper",
        "long_call": "Long call — directional bullish exposure (undefined upside)",
        "long_put": "Long put — directional bearish exposure (undefined downside)",
        "aggressive_daytrader_call": "0–3DTE long call — short-duration bullish sniper, time/% stop",
        "aggressive_daytrader_put": "0–3DTE long put — short-duration bearish sniper, time/% stop",
        "iron_condor": "Iron condor — premium collection in range-bound market",
        "long_iron_condor": "Long iron condor — owns breakout in range-bound market",
        "0dte_iron_condor": "0DTE iron condor — same-day premium harvest in range",
        "1dte_iron_condor": "1DTE iron condor — next-day premium harvest in range",
        "long_straddle": "Long straddle — owns move in either direction (vol play)",
        "scalp_long_straddle": "Short-dated straddle — owns intraday volatility move",
        "long_strangle": "Long strangle — owns breakout in either direction",
        "debit_spread_call": "Call debit spread — bullish directional defined-risk",
        "debit_spread_put": "Put debit spread — bearish directional defined-risk",
        "calendar_spread": "Calendar spread — transition regime, sells near / buys far expiry",
        "no_trade": "No trade — conditions not met for entry",
        "no_trade_or_micro_position": "No trade / micro probe — direction too ambiguous",
        "aggressive_daytrader_0DTE_straddle": "0DTE straddle — high-vol event gamma capture",
    }
    if not strategy_name:
        return "Unknown"
    return _MAP.get(strategy_name, strategy_name.replace("_", " ").title())


def _format_matched_legs(matched_legs: list, combo_qty: int) -> str:
    """Format option legs into a compact human-readable string."""
    if not matched_legs:
        return ""
    parts = []
    for leg in matched_legs:
        side = str(leg.get("side") or "").upper()
        opt_type = str(leg.get("option_type") or "").upper()
        strike = leg.get("strike")
        expiry = str(leg.get("expiry") or "?")
        leg_qty = leg.get("quantity") or combo_qty or 1
        strike_fmt = f"{float(strike):g}" if strike is not None else "?"
        type_char = opt_type[0] if opt_type else "?"
        parts.append(f"{side} {leg_qty}x {strike_fmt}{type_char} {expiry}")
    return "  |  ".join(parts)


def _plan_by_pid(plans_data: dict) -> dict:
    """Build a dict mapping plan_id -> plan row from the universe_trade_plans payload."""
    result: dict[str, dict] = {}
    for row in plans_data.get("results") or []:
        pid = str(row.get("plan_id") or "")
        if pid:
            result[pid] = row
    return result


def _fmt_dollar(v: Any) -> str:
    try:
        fv = float(v)
        return f"${fv:,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pnl_row(row: dict) -> str:
    """Format unrealized_pnl + unrealized_pnl_pct from a trade_log row."""
    upnl = row.get("unrealized_pnl") or ""
    upnl_pct = row.get("unrealized_pnl_pct") or ""
    try:
        fv = float(upnl)
        dollar_str = f"+${fv:,.2f}" if fv >= 0 else f"-${abs(fv):,.2f}"
    except (TypeError, ValueError):
        dollar_str = str(upnl)
    try:
        pct_str = f"{float(upnl_pct):+.1f}%"
    except (TypeError, ValueError):
        pct_str = str(upnl_pct)
    if dollar_str and pct_str:
        return f"{dollar_str}  ({pct_str})"
    return dollar_str or pct_str


_SEP = "─" * 30

# User-facing labels for the three parallel paper sleeves.
ACCOUNT_LARGE_OPTIONS = "Account: LARGE OPTIONS (local paper book · not IBKR)"
ACCOUNT_LARGE_EQUITIES = "Account: LARGE EQUITIES (IBKR paper · stocks only)"
ACCOUNT_PDT_CHALLENGE = "Account: PDT CHALLENGE ($1K→$25K · local paper)"


def _large_options_book_status(root: Path) -> str:
    """One line: seeded book value (for footers on entry / TP)."""
    try:
        snap = options_book_snapshot(root)
        return (
            f"Large-options book: {_fmt_dollar(snap.book_value)} "
            f"(seed {_fmt_dollar(snap.seed)}; closed realized + open MTM)"
        )
    except Exception:  # noqa: BLE001
        return "Large-options book: (unavailable)"


def _large_equities_book_status(root: Path) -> str:
    try:
        snap = equity_book_snapshot(root)
        return (
            f"Large-equities book: {_fmt_dollar(snap.book_value)} "
            f"(seed {_fmt_dollar(snap.seed)}; closed realized + open MTM)"
        )
    except Exception:  # noqa: BLE001
        return "Large-equities book: (unavailable)"


def _plan_option_structure_lines(plan: dict, row: dict[str, str] | None) -> tuple[str, str, int]:
    """Human strategy label, one-line structure (legs or fallback), combo qty."""
    decision = plan.get("decision") or {}
    row = row or {}
    strategy_name = str(decision.get("strategy_name") or plan.get("strategy") or row.get("strategy", ""))
    human = _strategy_entry_human(strategy_name) if strategy_name else "—"
    matched_legs = plan.get("matched_legs") or []
    spec = plan_combo_spec(plan)
    combo_qty = int((spec or {}).get("quantity") or 1)
    legs_str = _format_matched_legs(matched_legs, combo_qty)
    structure = legs_str if legs_str else strategy_name or str(row.get("strategy", "")) or "—"
    return human, structure, combo_qty


def _options_exit_account_impact(root: Path) -> str:
    """One line summarizing large-options book after a closed row is on disk."""
    try:
        snap = options_book_snapshot(root)
        return (
            f"Book impact: large-options est. value {_fmt_dollar(snap.book_value)} "
            f"(seed {_fmt_dollar(snap.seed)}; closed realized + open MTM in monitor CSV)."
        )
    except Exception:  # noqa: BLE001
        return "Book impact: (could not read large-options book snapshot)"


def _equity_exit_pnl_usd(pdat: dict[str, Any]) -> float | None:
    try:
        ep = float(pdat.get("entry_price") or 0.0)
        xp = float(pdat.get("exit_price") or 0.0)
        qty = int(float(pdat.get("quantity") or 0))
    except (TypeError, ValueError):
        return None
    if qty <= 0 or xp <= 0:
        return None
    side = str(pdat.get("side") or "long").lower()
    if side == "short":
        return (ep - xp) * float(qty)
    return (xp - ep) * float(qty)


def _equity_exit_account_impact(root: Path) -> str:
    try:
        snap = equity_book_snapshot(root)
        return (
            f"Book impact: large-equities est. value {_fmt_dollar(snap.book_value)} "
            f"(seed {_fmt_dollar(snap.seed)}; from equity CSV + open marks)."
        )
    except Exception:  # noqa: BLE001
        return "Book impact: (could not read large-equities book snapshot)"


def _challenge_exit_reason_human(reason: str) -> str:
    return {
        "target": "profit target",
        "stop": "stop loss",
        "expiry": "expiry / DTE",
        "manual": "manual / engine",
    }.get(reason, reason)


def _build_challenge_entry_message(pos: dict[str, Any], state: dict[str, Any]) -> str:
    bal = state.get("balance")
    lines = [
        f"🟢 PDT CHALLENGE — NEW POSITION — {pos.get('symbol', '?')}",
        ACCOUNT_PDT_CHALLENGE,
        _SEP,
        f"Id:        {pos.get('position_id', '?')}",
        f"Structure: {pos.get('option_type', '?')} {pos.get('direction', '?')}  "
        f"strike {_fmt_dollar(pos.get('strike', ''))}  ×{pos.get('qty', '')}  "
        f"DTE {pos.get('dte_remaining', pos.get('dte_at_entry', ''))}",
        f"Entry:     premium {_fmt_dollar(pos.get('premium_per_share', ''))}/sh  "
        f"(cost {_fmt_dollar(pos.get('total_cost', ''))})",
        _SEP,
        f"Challenge balance: {_fmt_dollar(bal)}",
    ]
    return "\n".join(lines)


def _build_challenge_exit_message(trade: dict[str, Any]) -> str:
    pnl = trade.get("pnl", 0)
    bb = trade.get("balance_before")
    ba = trade.get("balance_after")
    reason = _challenge_exit_reason_human(str(trade.get("exit_reason", "")))
    lines = [
        f"✅ PDT CHALLENGE — CLOSED — {trade.get('symbol', '?')}",
        ACCOUNT_PDT_CHALLENGE,
        _SEP,
        f"Trade P&L:      {_fmt_pnl_row({'unrealized_pnl': pnl, 'unrealized_pnl_pct': trade.get('pnl_pct', '')})}",
        f"Balance:        {_fmt_dollar(bb)}  →  {_fmt_dollar(ba)}",
        f"Exit reason:    {reason}",
        _SEP,
        f"Strike {_fmt_dollar(trade.get('strike', ''))}  {trade.get('option_type', '')}  {trade.get('direction', '')}",
    ]
    return "\n".join(lines)


def _build_new_opt_message(
    sym: str,
    pid: str,
    mark: str,
    entry_debit: str,
    sig: str,
    dte_val: str,
    plan: dict,
    row: dict[str, str],
    root: Path,
) -> str:
    decis = plan.get("decision") or {}
    regime_key = str(plan.get("regime_key") or decis.get("regime_key") or "")
    rationale = str(decis.get("rationale") or "")
    strat_human, structure, combo_qty = _plan_option_structure_lines(plan, row)
    thresholds = plan.get("thresholds") or {}
    v_tp = thresholds.get("v_take_profit")
    v_stop = thresholds.get("v_hard_stop")
    candidate = plan.get("candidate") or {}
    target_pct = candidate.get("target_profit_pct")
    max_risk_pct = candidate.get("max_risk_pct")
    dte_min = candidate.get("target_dte_min")
    dte_max = candidate.get("target_dte_max")

    try:
        ed = float(entry_debit)
        total_debit = ed * float(combo_qty)
        entry_line = (
            f"Entry:     {_fmt_dollar(ed)} per combo  ×{combo_qty}  "
            f"≈ {_fmt_dollar(total_debit)} total debit (monitor basis)"
        )
    except (TypeError, ValueError):
        entry_line = f"Entry:     {_fmt_dollar(entry_debit)} per combo  ×{combo_qty}"

    lines = [
        f"🟢 LARGE OPTIONS — NEW POSITION — {sym}",
        ACCOUNT_LARGE_OPTIONS,
        _SEP,
        f"Id:        {pid}",
        f"Strategy:  {strat_human}",
        f"Structure: {structure}",
        entry_line,
    ]
    if dte_val:
        lines.append(f"DTE:       {dte_val} days (row)")
    if regime_key:
        lines.append(f"Regime:    {_regime_human(regime_key)}")
    if rationale and rationale not in (strat_human, structure):
        lines.append(f"Logic:     {rationale}")

    lines.append(_SEP)

    if v_tp is not None:
        tp_line = f"Profit:    mark ≥ {_fmt_dollar(v_tp)}"
        if target_pct is not None:
            tp_line += f"  (+{float(target_pct) * 100:.0f}% on debit)"
        lines.append(tp_line)
    elif target_pct is not None:
        lines.append(f"Profit:    target +{float(target_pct) * 100:.0f}% on debit paid")

    if v_stop is not None:
        lines.append(f"Hard stop: mark ≤ {_fmt_dollar(v_stop)}  (exit to cut loss)")

    exit_conds: list[str] = []
    if v_tp is not None:
        exit_conds.append(f"take profit ≥ {_fmt_dollar(v_tp)}")
    if v_stop is not None:
        exit_conds.append(f"hard stop ≤ {_fmt_dollar(v_stop)}")
    if dte_min is not None:
        exit_conds.append(f"time stop near {dte_min} DTE")
    if max_risk_pct is not None:
        exit_conds.append(f"max portfolio risk {float(max_risk_pct) * 100:.1f}%")
    if exit_conds:
        lines.append(f"Exit when: {' | '.join(exit_conds)}")

    if dte_min is not None and dte_max is not None:
        lines.append(f"Window:    {dte_min}–{dte_max} DTE  (aim to profit before ~{dte_min} DTE)")
    elif dte_max is not None:
        lines.append(f"Max DTE:   {dte_max} days")

    lines.append(_SEP)
    lines.append(f"Mark: {mark}  |  Signal: {sig}")
    lines.append(_SEP)
    lines.append(_large_options_book_status(root))
    return "\n".join(lines)


def _build_tp_opt_message(
    sym: str, pid: str, mark: str, row: dict, plan: dict, root: Path
) -> str:
    entry_debit = row.get("entry_debit") or row.get("entry_mid") or "?"
    dte_val = row.get("dte") or ""
    thresholds = plan.get("thresholds") or {}
    v_tp = thresholds.get("v_take_profit")
    strat_human, structure, combo_qty = _plan_option_structure_lines(plan, row)

    lines = [
        f"🎯 LARGE OPTIONS — PROFIT TARGET — {sym}",
        ACCOUNT_LARGE_OPTIONS,
        _SEP,
        f"Id:        {pid}",
        f"Strategy:  {strat_human}",
        f"Structure: {structure}",
        f"Entry:     {_fmt_dollar(entry_debit)}  (×{combo_qty} combo)",
        f"Mark now:  {mark}",
        f"P&L:       {_fmt_pnl_row(row)}",
    ]
    if v_tp is not None:
        lines.append(f"Target:    mark ≥ {_fmt_dollar(v_tp)}  ✓ reached")
    if dte_val:
        lines.append(f"DTE:       {dte_val} days remaining")
    lines.append(_SEP)
    lines.append(_large_options_book_status(root))
    lines.append("▶ Consider exiting — profit target reached")
    return "\n".join(lines)


def _build_exit_opt_message(sym: str, pid: str, mark: str, sig: str, row: dict, plan: dict) -> str:
    entry_debit = row.get("entry_debit") or row.get("entry_mid") or "?"
    dte_val = row.get("dte") or ""
    strat_human, structure, combo_qty = _plan_option_structure_lines(plan, row)

    if sig == "take_profit":
        emoji = "✅"
    elif sig in ("hard_stop", "max_loss_stop"):
        emoji = "🛑"
    elif sig in ("expiry_force_close", "time_stop"):
        emoji = "⏱"
    else:
        emoji = "🔴"

    lines = [
        f"{emoji} LARGE OPTIONS — CLOSED — {sym}",
        ACCOUNT_LARGE_OPTIONS,
        _SEP,
        f"Id:        {pid}",
        f"Strategy:  {strat_human}",
        f"Structure: {structure}",
        f"Entry was: {_fmt_dollar(entry_debit)}  (×{combo_qty} combo)",
        _SEP,
        f"Reason:    {_exit_reason_human(sig)}",
        f"Exit mark: {mark}",
        f"Final P&L: {_fmt_pnl_row(row)}",
    ]
    if dte_val:
        lines.append(f"DTE:       {dte_val} days at exit")
    return "\n".join(lines)


def _build_new_equity_message(root: Path, plan_id: str, pdat: dict[str, Any], plan: dict) -> str:
    sym = str(pdat.get("symbol", "?"))
    side = str(pdat.get("side", "?"))
    qty_raw = pdat.get("quantity", "")
    direction = str(pdat.get("direction", "") or "")
    entry_rk = str(pdat.get("entry_regime_key", "") or "")
    decision = plan.get("decision") or {}
    regime_key = str(plan.get("regime_key") or decision.get("regime_key") or entry_rk or "")
    rationale = str(decision.get("rationale") or "")
    strategy_name = str(decision.get("strategy_name") or plan.get("strategy") or "")
    strat_human = _strategy_entry_human(strategy_name) if strategy_name else "—"
    structure = f"{side.upper()} {qty_raw} sh"
    if direction:
        structure += f"  ·  thesis bias: {direction}"

    lines = [
        f"🟢 LARGE EQUITIES — NEW POSITION — {sym}",
        ACCOUNT_LARGE_EQUITIES,
        _SEP,
        f"Id:        {plan_id}",
        f"Strategy:  {strat_human}",
        f"Structure: {structure}",
    ]
    try:
        ep = float(pdat.get("entry_price") or 0.0)
        qi = int(float(qty_raw))
        notion = ep * float(qi)
        lines.append(f"Entry:     {_fmt_dollar(ep)}/sh  (est. notional {_fmt_dollar(notion)})")
    except (TypeError, ValueError):
        lines.append(f"Entry:     {_fmt_dollar(pdat.get('entry_price', ''))}  qty {qty_raw}")

    if regime_key:
        lines.append(f"Regime:    {_regime_human(regime_key)}")
    if rationale and rationale != strategy_name:
        lines.append(f"Logic:     {rationale}")

    lines.extend([_SEP, _large_equities_book_status(root)])
    return "\n".join(lines)


def _build_exit_equity_message(root: Path, plan_id: str, pdat: dict[str, Any], exit_reason: str) -> str:
    sym = str(pdat.get("symbol", "?"))
    side = str(pdat.get("side", "?"))
    qty_raw = pdat.get("quantity", "")
    lines_eq: list[str] = [
        f"🔴 LARGE EQUITIES — CLOSED — {sym}",
        ACCOUNT_LARGE_EQUITIES,
        _SEP,
        f"Id:        {plan_id}",
        f"Structure: {side.upper()} {qty_raw} sh",
        _SEP,
        f"Reason:    {exit_reason}",
    ]
    try:
        ep = float(pdat.get("entry_price") or 0.0)
        xp = float(pdat.get("exit_price") or 0.0)
        lines_eq.append(f"Entry:     {_fmt_dollar(ep)}/sh  →  exit {_fmt_dollar(xp)}/sh")
    except (TypeError, ValueError):
        pass
    pnl_est = _equity_exit_pnl_usd(pdat)
    if pnl_est is not None:
        lines_eq.append(f"This exit P&L (est.): {_fmt_dollar(pnl_est)}")
    lines_eq.extend([_SEP, _equity_exit_account_impact(root)])
    return "\n".join(lines_eq)


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


def _challenge_expiry_from_entry(entry_date: str, dte_at_entry: Any) -> date | None:
    """Expiry ≈ entry session date + ``dte_at_entry`` calendar days (matches challenge engine)."""
    try:
        d0 = date.fromisoformat(str(entry_date).strip()[:10])
        dte = int(float(dte_at_entry))
        return d0 + timedelta(days=dte)
    except (ValueError, TypeError):
        return None


def _positions_challenge_section(root: Path, *, max_positions: int) -> list[str]:
    """Open PDT challenge positions + balance (``data/challenge/state.json``)."""
    ch_path = default_paths(root)["challenge_state"]
    lines: list[str] = [
        "─── PDT CHALLENGE ($1K→$25K · local paper) ───",
        f"    {ACCOUNT_PDT_CHALLENGE}",
    ]
    if not ch_path.is_file():
        lines.append("    (no challenge state — `rlm challenge --reset`)")
        return lines
    try:
        raw = json.loads(ch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lines.append("    (unreadable challenge state)")
        return lines
    bal = float(raw.get("balance", 0))
    seed = float(raw.get("seed", 1000))
    opens = [x for x in (raw.get("open_positions") or []) if isinstance(x, dict)]
    lines.append(f"    Cash: {_fmt_dollar(bal)}  seed {_fmt_dollar(seed)}  ·  {len(opens)} open contract leg(s)")
    lu = str(raw.get("last_updated") or "").strip() or "?"
    mtime_s = (
        datetime.fromtimestamp(ch_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if ch_path.is_file()
        else "?"
    )
    lines.append(f"    State: last_updated={lu}  ·  state.json mtime {mtime_s}")
    lines.append(
        "    Note: contract marks & MTM refresh when a challenge session runs "
        "(`rlm challenge --run` or your master/everything loop), not on every Telegram poll."
    )
    if not opens:
        lines.append("    (no open challenge positions)")
        return lines
    sorted_o = sorted(opens, key=lambda x: (str(x.get("symbol") or ""), str(x.get("position_id") or "")))
    for pos in sorted_o[:max_positions]:
        pid = pos.get("position_id", "?")
        sym = str(pos.get("symbol") or "?")
        opt = pos.get("option_type", "?")
        direc = str(pos.get("direction") or "?")
        strike = pos.get("strike", "")
        qty = pos.get("qty", "")
        entry = pos.get("entry_date", "?")
        dte_ent = pos.get("dte_at_entry", "")
        dte = pos.get("dte_remaining", dte_ent)
        upnl = pos.get("unrealised_pnl", 0)
        cost = pos.get("total_cost", "")
        prem_e = pos.get("premium_per_share", "")
        prem_c = pos.get("current_premium", prem_e)
        exp_d = _challenge_expiry_from_entry(str(entry), dte_ent)
        occ = "?"
        exp_txt = "expiry ?"
        if exp_d is not None:
            exp_txt = exp_d.isoformat()
            try:
                occ = format_occ_compact_symbol(sym, exp_d, option_type=str(opt), strike=float(strike))
            except (TypeError, ValueError):
                occ = f"{sym} ?"
        cp_lbl = "C" if str(opt).lower().startswith("c") else "P"
        lines.append(
            f"  • {occ}  ·  {exp_txt}  ·  {direc.upper()} {cp_lbl} ×{qty} @ K={_fmt_dollar(strike)}"
        )
        lines.append(
            f"    challenge_id={pid}  opened {entry}  DTE_now={dte}  "
            f"mark {_fmt_dollar(prem_c)}/sh (entry {_fmt_dollar(prem_e)}/sh)  "
            f"cost {_fmt_dollar(cost)}  MTM {_fmt_dollar(upnl)}"
        )
    if len(opens) > max_positions:
        lines.append(f"    … {len(opens) - max_positions} more")
    return lines


def build_universe_and_positions(root: Path, *, max_active: int = 12, max_positions: int = 20) -> str:
    """Positions grouped by trading account (large options, large equities, PDT); then active universe."""
    p = default_paths(root)
    plans_data = _read_plans(p["plans"])
    plan_lookup = _plan_by_pid(plans_data)
    univ_text = (
        build_universe_report_from_data(plans_data, max_active=max_active)
        if plans_data
        else build_universe_report(root, max_active=max_active)
    )

    lines: list[str] = ["=== Positions by account ===", ""]

    latest = _latest_rows_per_plan_csv(p["trade_log"])
    opts: list[tuple[str, dict[str, str]]] = []
    for pid, row in latest.items():
        if (row.get("closed") or "0").strip() != "1":
            opts.append((pid, row))
    opts.sort(key=lambda t: (str(t[1].get("symbol") or ""), t[0]))

    lines.extend(
        [
            "─── LARGE OPTIONS (local paper book · not IBKR) ───",
            f"    {ACCOUNT_LARGE_OPTIONS}",
            f"    Open: {len(opts)} monitor position(s) (trade_log)",
        ]
    )
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

            plan = plan_lookup.get(pid, {})
            strat = _universe_row_strategy(plan)
            if len(strat) > 40:
                strat = strat[:37] + "…"
            _, struct, cq = _plan_option_structure_lines(plan, row)
            ed = row.get("entry_debit") or row.get("entry_mid") or ""
            leg_line = ""
            if struct and struct != strat and struct != "—":
                leg_line = struct
                if len(leg_line) > 100:
                    leg_line = leg_line[:97] + "…"

            lines.append(
                f"  • {row.get('symbol', '?')}  {pid}\n"
                f"      strategy={strat}  debit≈{_fmt_dollar(ed)} ×{cq}  mark={row.get('current_mark', '')}  "
                f"PnL={pnl_fmt}  {row.get('signal', '')}  DTE={row.get('dte', '')}{warn_suffix}"
            )
            if leg_line:
                lines.append(f"      legs: {leg_line}")

        if len(opts) > max_positions:
            lines.append(f"  … {len(opts) - max_positions} more")

    eq = _read_equity_state(p["equity_state"])
    eq_open = [(str(k), v or {}) for k, v in eq.items() if str((v or {}).get("status") or "") == "open"]
    eq_open.sort(key=lambda t: (str(t[1].get("symbol") or ""), t[0]))
    eq_log = _latest_equity_open_rows_by_plan(p["equity_trade_log"])

    lines.extend(
        [
            "",
            "─── LARGE EQUITIES (IBKR paper · stocks only) ───",
            f"    {ACCOUNT_LARGE_EQUITIES}",
            f"    Open: {len(eq_open)} stock position(s)",
        ]
    )
    if not eq_open:
        lines.append("  (none open)")
    else:
        for pid, d in eq_open[:max_positions]:
            lr = eq_log.get(pid, {})
            sym = d.get("symbol", "?")
            side = str(d.get("side", "?"))
            qty = d.get("quantity", "")
            ep = d.get("entry_price", "")
            mark = lr.get("current_mark", "") if lr else ""
            usd_raw = lr.get("unrealized_pnl", "") if lr else ""
            upct_raw = lr.get("unrealized_pnl_pct", "") if lr else ""
            sig = lr.get("signal", "") if lr else ""
            eq_plan = plan_lookup.get(pid, {})
            thesis = _universe_row_strategy(eq_plan)
            if len(thesis) > 36:
                thesis = thesis[:33] + "…"
            reg_h = _universe_row_regime_head(eq_plan)
            pnl_usd_try = ""
            try:
                if str(usd_raw).strip():
                    fu = float(usd_raw)
                    pnl_usd_try = f"  ${fu:+,.2f}"
            except (TypeError, ValueError):
                pass
            pct_try = ""
            try:
                if str(upct_raw).strip():
                    fp = float(upct_raw)
                    pct_try = f" ({fp:+.2f}%)"
            except (TypeError, ValueError):
                pct_try = f" ({upct_raw})" if upct_raw else ""

            lines.append(
                f"  • {sym}  {pid}\n"
                f"      {side.upper()} {qty} sh @ {_fmt_dollar(ep)}  mark={mark}{pnl_usd_try}{pct_try}  "
                f"{sig}  thesis={thesis}  regime={reg_h}"
            )

        if len(eq_open) > max_positions:
            lines.append(f"  … {len(eq_open) - max_positions} more")

    lines.append("")
    lines.extend(_positions_challenge_section(root, max_positions=max_positions))

    lines.extend(
        [
            "",
            "=== Active universe (plan source — not a separate balance) ===",
            univ_text,
        ]
    )

    return "\n".join(lines)


def _active_plan_ids_from_plans(data: dict[str, Any]) -> set[str]:
    """Active ``plan_id`` set (aligned with equity monitor + ranked/results union)."""
    return _active_plan_ids_from_plans_payload(data)


def _universe_row_strategy(r: dict[str, Any]) -> str:
    d = r.get("decision") if isinstance(r.get("decision"), dict) else {}
    s = str(d.get("strategy_name") or "").strip()
    if s:
        return s
    return str(r.get("strategy") or "").strip() or "—"


def _universe_row_confidence_pct(r: dict[str, Any]) -> str:
    for key in ("regime_confidence", "confidence"):
        raw = r.get(key)
        if raw is not None and str(raw) not in ("", "nan", "None"):
            try:
                v = float(raw)
                return f"{v * 100:.1f}%" if abs(v) <= 1.0 else f"{v:.1f}%"
            except (TypeError, ValueError):
                return str(raw).strip()
    d = r.get("decision") if isinstance(r.get("decision"), dict) else {}
    meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
    for key in ("regime_confidence", "confidence", "kronos_confidence"):
        raw = meta.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
            return f"{v * 100:.1f}%" if abs(v) <= 1.0 else f"{v:.1f}%"
        except (TypeError, ValueError):
            continue
    pl = r.get("pipeline") if isinstance(r.get("pipeline"), dict) else {}
    raw = pl.get("kronos_confidence")
    if raw is not None:
        try:
            v = float(raw)
            return f"{v * 100:.1f}%"
        except (TypeError, ValueError):
            pass
    return "—"


def _universe_row_regime_head(r: dict[str, Any]) -> str:
    d = r.get("decision") if isinstance(r.get("decision"), dict) else {}
    rk = str(r.get("regime_key") or d.get("regime_key") or "").strip()
    if not rk:
        return "—"
    return rk.split("|", 1)[0].strip() or "—"


def _latest_equity_open_rows_by_plan(path: Path) -> dict[str, dict[str, str]]:
    """Last open (closed!=1) row per plan_id from equity_trade_log."""
    rows = _load_all_csv_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = str(row.get("plan_id") or "")
        if not pid:
            continue
        if (row.get("closed") or "0").strip() == "1":
            continue
        out[pid] = row
    return out


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
    actives: list[dict[str, Any]] = list(_iter_active_trade_plan_rows(data))
    actives.sort(key=lambda x: float(x.get("rank_score") or 0.0), reverse=True)
    lines = [
        f"Universe report (top {min(max_active, len(actives))} of {len(actives)} active)\n" f"generated_at: {gen}\n"
    ]
    for r in actives[:max_active]:
        sym = r.get("symbol", "?")
        st = _universe_row_strategy(r)
        if len(st) > 44:
            st = st[:41] + "…"
        rs = r.get("rank_score")
        conf_fmt = _universe_row_confidence_pct(r)
        reg_h = _universe_row_regime_head(r)
        pid = r.get("plan_id", "")
        try:
            rs_fmt = f"{float(rs):.4f}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            rs_fmt = str(rs or "?")
        lines.append(
            f"  • {sym}  strategy={st}  regime={reg_h}  score={rs_fmt}  conf={conf_fmt}  id={pid}"
        )
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
        return "--- PDT (Robinhood $1K → $25K · elite paper track) ---\n  (no challenge state — run `rlm challenge --reset`)"

    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "--- PDT (Robinhood $1K → $25K · elite paper track) ---\n  (unreadable state file)"

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
        "--- PDT (Robinhood $1K → $25K · elite paper track) ---",
        f"Balance:   ${balance:,.2f}",
        f"Seed:      ${seed:,.2f}",
        f"Progress:  {progress:.1f}% ({milestone_label})",
        f"Today (realized, exit date):     {_fmt_pnl(daily)}",
        f"This week (realized, exit date): {_fmt_pnl(weekly)}",
        f"Net vs seed (cash):              {_fmt_pnl(all_time)}",
        f"Open MTM (unrealized):           {_fmt_pnl(open_mtm)}",
        f"Ledger:    data/processed/ledgers/pdt_robinhood_challenge_book.csv",
    ]
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
    ledger_note = ""
    try:
        write_trading_ledgers(root)
        ledger_note = (
            "\n--- LEDGERS (CSV for Sheets/Excel) ---\n"
            "  data/processed/ledgers/large_equities_book.csv\n"
            "  data/processed/ledgers/large_options_book.csv\n"
            "  data/processed/ledgers/pdt_robinhood_challenge_book.csv\n"
        )
    except Exception as exc:  # noqa: BLE001
        ledger_note = f"\n(ledgers sync error: {exc})\n"

    p = default_paths(root)
    sections: list[str] = ["=== P&L REPORT ==="]

    eq_rows = _load_all_csv_rows(p["equity_trade_log"])
    eq_d, eq_w, eq_a, eq_mtm = _pnl_aggregates_from_log(eq_rows)
    eq_snap = equity_book_snapshot(root)
    eq_net = eq_snap.book_value - eq_snap.seed
    sections.append(
        "\n".join(
            [
                "",
                "--- LARGE EQUITIES (IBKR · prop-style book) ---",
                f"Book seed: ${eq_snap.seed:,.2f}   Book value: ${eq_snap.book_value:,.2f}",
                f"Net vs seed: {_fmt_pnl(eq_net)}  "
                f"(realized on closes {_fmt_pnl(eq_snap.closed_realized)} + open MTM {_fmt_pnl(eq_snap.open_mtm)})",
                f"Today (realized, exit date):     {_fmt_pnl(eq_d)}",
                f"This week (realized, exit date): {_fmt_pnl(eq_w)}",
                f"Realized all-time (closed only): {_fmt_pnl(eq_a)}",
                f"Open MTM (unrealized):           {_fmt_pnl(eq_mtm)}",
            ]
        )
    )

    opt_rows = _load_all_csv_rows(p["trade_log"])
    opt_d, opt_w, opt_a, opt_mtm = _pnl_aggregates_from_log(opt_rows)
    opt_snap = options_book_snapshot(root)
    opt_net = opt_snap.book_value - opt_snap.seed
    sections.append(
        "\n".join(
            [
                "",
                "--- LARGE OPTIONS (advanced book · local monitor / not IBKR) ---",
                f"Book seed: ${opt_snap.seed:,.2f}   Book value: ${opt_snap.book_value:,.2f}",
                f"Net vs seed: {_fmt_pnl(opt_net)}  "
                f"(realized on closes {_fmt_pnl(opt_snap.closed_realized)} + open MTM {_fmt_pnl(opt_snap.open_mtm)})",
                f"Today (realized, exit date):     {_fmt_pnl(opt_d)}",
                f"This week (realized, exit date): {_fmt_pnl(opt_w)}",
                f"Realized all-time (closed only): {_fmt_pnl(opt_a)}",
                f"Open MTM (unrealized):           {_fmt_pnl(opt_mtm)}",
            ]
        )
    )

    sections.append("")
    sections.append(_challenge_pnl_section(root))

    sections.append("")
    sections.append(_ibkr_balance_section())

    return "\n".join(sections) + ledger_note


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
    challenge_trade_n: int = 0
    challenge_open_ids: set[str] = field(default_factory=set)

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
        try:
            s.challenge_trade_n = int(d.get("challenge_trade_n", 0))
        except (TypeError, ValueError):
            s.challenge_trade_n = 0
        raw_ch = d.get("challenge_open_ids")
        if raw_ch is not None:
            s.challenge_open_ids = {str(x) for x in raw_ch}
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
            "challenge_trade_n": self.challenge_trade_n,
            "challenge_open_ids": sorted(self.challenge_open_ids),
        }


def _challenge_notification_messages(root: Path, st: _St) -> list[str]:
    """Emit Telegram lines for new challenge opens / closes; updates ``st`` challenge counters."""
    msgs: list[str] = []
    path = default_paths(root)["challenge_state"]
    if not path.is_file():
        st.challenge_trade_n = 0
        st.challenge_open_ids = set()
        return msgs
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return msgs
    if not isinstance(raw, dict):
        return msgs
    history = list(raw.get("trade_history") or [])
    opens = list(raw.get("open_positions") or [])
    cur_open_ids = {str(o.get("position_id")) for o in opens if o.get("position_id")}
    prev_n = st.challenge_trade_n
    prev_open = st.challenge_open_ids

    if len(history) > prev_n:
        for t in history[prev_n:]:
            if isinstance(t, dict):
                msgs.append(_build_challenge_exit_message(t))

    for pos in opens:
        if not isinstance(pos, dict):
            continue
        pid = str(pos.get("position_id") or "")
        if pid and pid not in prev_open:
            msgs.append(_build_challenge_entry_message(pos, raw))

    st.challenge_trade_n = len(history)
    st.challenge_open_ids = cur_open_ids
    return msgs


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

    # Load plans data once up front so all alert builders can use it
    plans_data = _read_plans(p["plans"])
    plan_lookup = _plan_by_pid(plans_data)

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
        st.last_universe_active_ids = _active_plan_ids_from_plans(plans_data)
        ch_path = p["challenge_state"]
        if ch_path.is_file():
            try:
                ch_raw = json.loads(ch_path.read_text(encoding="utf-8"))
                st.challenge_trade_n = len(ch_raw.get("trade_history") or [])
                st.challenge_open_ids = {
                    str(o.get("position_id"))
                    for o in (ch_raw.get("open_positions") or [])
                    if o.get("position_id")
                }
            except (OSError, json.JSONDecodeError):
                st.challenge_trade_n = 0
                st.challenge_open_ids = set()
        else:
            st.challenge_trade_n = 0
            st.challenge_open_ids = set()
        st.notify_seeded = True
        merged = {**state_blob, **st.to_json()}
        return [], merged

    # Upgrade older state files that used known_option_plans but not announced_trade_open
    if st.notify_seeded and "announced_trade_open" not in state_blob:
        st.announced_trade_open = {pid for pid, row in latest.items() if (row.get("closed") or "0").strip() != "1"}
        merged = {**state_blob, **st.to_json()}
        return [], merged

    if st.notify_seeded and "last_universe_active_ids" not in state_blob:
        st.last_universe_active_ids = _active_plan_ids_from_plans(plans_data)
        merged = {**state_blob, **st.to_json()}
        return [], merged

    if st.notify_seeded and "challenge_trade_n" not in state_blob:
        ch_path = p["challenge_state"]
        if ch_path.is_file():
            try:
                ch_raw = json.loads(ch_path.read_text(encoding="utf-8"))
                st.challenge_trade_n = len(ch_raw.get("trade_history") or [])
                st.challenge_open_ids = {
                    str(o.get("position_id"))
                    for o in (ch_raw.get("open_positions") or [])
                    if o.get("position_id")
                }
            except (OSError, json.JSONDecodeError):
                st.challenge_trade_n = 0
                st.challenge_open_ids = set()
        merged = {**state_blob, **st.to_json()}
        return [], merged

    for pid, row in latest.items():
        sig = (row.get("signal") or "hold").strip()
        mark = row.get("current_mark", "")
        sym = row.get("symbol", "")
        closed = (row.get("closed") or "0").strip() == "1"
        prev = st.last_opt_signal.get(pid, "")
        plan = plan_lookup.get(pid, {})

        if not closed and pid not in st.announced_trade_open:
            st.announced_trade_open.add(pid)
            ed = row.get("entry_debit", row.get("entry_mid", ""))
            dte_val = row.get("dte", "")
            out.append(_build_new_opt_message(sym, pid, mark, ed, sig, dte_val, plan, row, root))

        if sig == "take_profit" and prev != "take_profit" and pid not in st.announced_tp:
            st.announced_tp.add(pid)
            out.append(_build_tp_opt_message(sym, pid, mark, row, plan, root))

        if closed and sig in EXIT_SIGNALS and pid not in st.announced_exit:
            st.announced_exit.add(pid)
            st.announced_trade_open.discard(pid)
            out.append(
                _build_exit_opt_message(sym, pid, mark, sig, row, plan)
                + "\n"
                + _options_exit_account_impact(root)
            )

        st.last_opt_signal[pid] = sig

    for gone in set(st.announced_tp) - set(latest.keys()):
        st.announced_tp.discard(gone)
    for gone in set(st.announced_exit) - set(latest.keys()):
        st.announced_exit.discard(gone)
    for pid in list(st.announced_trade_open):
        row = latest.get(pid)
        if not row or (row.get("closed") or "0").strip() == "1":
            st.announced_trade_open.discard(pid)

    cur_u = _active_plan_ids_from_plans(plans_data)
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
                eq_plan = plan_lookup.get(pkey, {})
                out.append(_build_new_equity_message(root, pkey, pdat, eq_plan))
        elif st_eq == "closed" and pkey in prev_eq and pkey not in st.announced_equity_close:
            st.announced_equity_close.add(pkey)
            ex = pdat.get("exit_reason") or pdat.get("note") or "—"
            out.append(_build_exit_equity_message(root, pkey, pdat, str(ex)))

    st.last_equity_open = now_open
    out.extend(_challenge_notification_messages(root, st))
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
