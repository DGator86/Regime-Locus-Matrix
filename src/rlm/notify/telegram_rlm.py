"""
Telegram push logic driven by RLM on-disk state (no changes to the trading stack).

* Universe: new **active symbol** in ``universe_trade_plans.json`` (Robinhood BUY IDEA; plan_id
  timestamps rotate on every rescan and must not re-alert). Optional ``session_brief.json`` for /brief.
* Options monitor: ``trade_log.csv`` (open / take-profit / exit signals); leg detail is also read from
  ``data/processed/trade_plan_snapshots.json`` when a plan_id is no longer in ``universe_trade_plans.json``.
* Equities: ``equity_positions_state.json``.
* Balances: optional ``fetch_ibkr_account_snapshot`` (requires IB Gateway + ``ibapi``).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from rlm.execution.combo_spec import plan_combo_spec
from rlm.execution.exit_signals import EXIT_SIGNALS
from rlm.notify.ledger_books import (
    book_pnl_aggregates,
    equity_book_snapshot,
    load_equity_trade_log_rows,
    load_options_trade_log_rows,
    options_book_snapshot,
    write_trading_ledgers,
)
from rlm.notify.options_paths import options_trade_log_primary, options_trade_log_read_paths
from rlm.notify.options_plain_language import humanize_strategy_name as _strategy_entry_human
from rlm.universe.active_plans import active_plan_ids as _active_plan_ids_from_plans_payload
from rlm.universe.active_plans import active_symbols as _active_symbols_from_plans_payload
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
        "trade_plan_snapshots": root / "data" / "processed" / "trade_plan_snapshots.json",
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


def _plan_by_pid(plans_data: dict) -> dict[str, dict]:
    """Map plan_id -> plan row (``results`` first, then ``active_ranked`` for gaps)."""
    result: dict[str, dict] = {}
    for row in plans_data.get("results") or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("plan_id") or "")
        if pid:
            result[pid] = row
    for row in plans_data.get("active_ranked") or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("plan_id") or "")
        if pid and pid not in result:
            result[pid] = row
    return result


def _plan_row_rank_key(row: dict[str, Any]) -> tuple[int, int, float]:
    active = 1 if str(row.get("status") or "").strip().lower() == "active" else 0
    d = row.get("decision") if isinstance(row.get("decision"), dict) else {}
    rk = str(row.get("regime_key") or d.get("regime_key") or "").strip()
    has_regime = 1 if rk else 0
    try:
        rs = float(row.get("rank_score") or 0.0)
    except (TypeError, ValueError):
        rs = 0.0
    return (active, has_regime, rs)


def _plan_by_symbol(plans_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Symbol -> best universe row (active + regime preferred; equity plan_ids differ from options)."""
    out: dict[str, dict[str, Any]] = {}
    for row in (plans_data.get("active_ranked") or []) + (plans_data.get("results") or []):
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        prev = out.get(sym)
        if prev is None or _plan_row_rank_key(row) > _plan_row_rank_key(prev):
            out[sym] = row
    return out


def _equity_display_thesis(eq_plan: dict[str, Any], pdat: dict[str, Any], eq_log_row: dict[str, str]) -> str:
    thesis = _universe_row_strategy(eq_plan)
    if thesis != "—":
        return thesis
    direction = str(pdat.get("direction") or eq_log_row.get("strategy") or "").strip()
    if direction:
        return direction
    return "—"


def _equity_display_regime(eq_plan: dict[str, Any], pdat: dict[str, Any]) -> str:
    reg = _universe_row_regime_head(eq_plan)
    if reg != "—":
        return reg
    rk = str(pdat.get("entry_regime_key") or "").strip()
    if rk:
        return rk.split("|", 1)[0].strip() or "—"
    return "—"


def _resolved_equity_plan(
    plans_data: dict[str, Any],
    plan_lookup: dict[str, dict],
    plan_by_sym: dict[str, dict[str, Any]],
    plan_id: str,
    symbol: str,
) -> dict[str, Any]:
    pid = str(plan_id or "").strip()
    if pid and pid in plan_lookup:
        return plan_lookup[pid]
    sym = str(symbol or "").strip().upper()
    if sym and sym in plan_by_sym:
        return plan_by_sym[sym]
    return {}


def _load_trade_plan_snapshots(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _resolved_options_plan(
    plans_data: dict[str, Any],
    pid: str,
    snapshots: dict[str, dict],
) -> dict[str, Any]:
    """Universe plan merged with monitor snapshot (keeps legs after plan drops from JSON)."""
    base = _plan_by_pid(plans_data).get(pid, {})
    if not isinstance(base, dict):
        base = {}
    snap = snapshots.get(pid)
    if not isinstance(snap, dict):
        return dict(base)
    if not base:
        return dict(snap)
    merged: dict[str, Any] = dict(base)
    if not (merged.get("matched_legs") or []) and (snap.get("matched_legs") or []):
        merged["matched_legs"] = snap["matched_legs"]
    if plan_combo_spec(merged) is None and plan_combo_spec(snap) is not None:
        merged["combo_spec"] = snap.get("combo_spec")
    if not (merged.get("decision") or {}) and (snap.get("decision") or {}):
        merged["decision"] = snap["decision"]
    if not str(merged.get("strategy") or "").strip() and str(snap.get("strategy") or "").strip():
        merged["strategy"] = snap.get("strategy")
    if not str(merged.get("symbol") or "").strip() and str(snap.get("symbol") or "").strip():
        merged["symbol"] = snap.get("symbol")
    if not (merged.get("thresholds") or {}) and (snap.get("thresholds") or {}):
        merged["thresholds"] = snap["thresholds"]
    if merged.get("entry_debit_dollars") is None and snap.get("entry_debit_dollars") is not None:
        merged["entry_debit_dollars"] = snap.get("entry_debit_dollars")
    if merged.get("entry_mid_mark_dollars") is None and snap.get("entry_mid_mark_dollars") is not None:
        merged["entry_mid_mark_dollars"] = snap.get("entry_mid_mark_dollars")
    return merged


def _plan_with_trade_log_legs(plan: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    """Fill ``matched_legs`` from ``trade_log`` ``legs_json`` when universe/snapshots lack legs."""
    out = dict(plan)
    if out.get("matched_legs"):
        return out
    lj = (row.get("legs_json") or "").strip()
    if not lj:
        return out
    try:
        legs = json.loads(lj)
    except json.JSONDecodeError:
        return out
    if isinstance(legs, list) and legs:
        out["matched_legs"] = legs
    return out


def _legs_from_combo_spec_display(spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``combo_spec.legs`` as ``matched_legs``-shaped rows for Telegram."""
    if not spec or not isinstance(spec, dict):
        return []
    raw = spec.get("legs")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for leg in raw:
        if not isinstance(leg, dict):
            continue
        exp = str(leg.get("expiry") or "").strip()
        exp_iso = exp
        if len(exp) == 8 and exp.isdigit():
            exp_iso = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}"
        out.append(
            {
                "side": str(leg.get("side") or "long"),
                "option_type": str(leg.get("option_type") or "call"),
                "strike": leg.get("strike"),
                "expiry": exp_iso,
            }
        )
    return out


def _fmt_dollar(v: Any) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv < 0:
        return f"-${abs(fv):,.2f}"
    return f"${fv:,.2f}"


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
ACCOUNT_PDT_CHALLENGE = "Account: RLM CHALLENGE ($1K→$100K · cash account · local paper)"
ACCOUNT_ROBINHOOD = "Account: ROBINHOOD (manual buy · not auto-executed)"


def _notify_flag(name: str, *, default: str = "1") -> bool:
    raw = (os.environ.get(name) or default).strip().lower()
    return raw in ("1", "true", "yes", "on")


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
    matched_legs = list(plan.get("matched_legs") or [])
    if not matched_legs:
        matched_legs = _legs_from_combo_spec_display(plan_combo_spec(plan))
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


def _plan_pay_and_exit_lines(plan: dict[str, Any]) -> list[str]:
    """Debit to pay and projected combo mark exits from pipeline thresholds."""
    thresholds = plan.get("thresholds") or {}
    candidate = plan.get("candidate") or {}
    spec = plan_combo_spec(plan) or {}
    combo_qty = max(1, int(spec.get("quantity") or 1))

    pay: float | None = None
    for key in ("entry_debit_dollars", "entry_mid_mark_dollars"):
        try:
            val = float(plan.get(key) or 0.0)
            if val > 0:
                pay = val
                break
        except (TypeError, ValueError):
            continue
    if pay is None:
        try:
            lim = float(spec.get("limit_price") or 0.0)
            if lim > 0:
                pay = lim * 100.0 * combo_qty
        except (TypeError, ValueError):
            pay = None

    lines: list[str] = []
    if pay is not None:
        per_combo = pay / float(combo_qty)
        lines.append(
            f"Pay (est.):  {_fmt_dollar(per_combo)} per combo  ×{combo_qty}  "
            f"≈ {_fmt_dollar(pay)} total debit"
        )

    v_tp = thresholds.get("v_take_profit")
    v_stop = thresholds.get("v_hard_stop")
    v0 = thresholds.get("v0") or plan.get("entry_mid_mark_dollars")
    if v_tp is not None:
        tp_line = f"Exit target: {_fmt_dollar(v_tp)} combo mark (take profit)"
        target_pct = candidate.get("target_profit_pct")
        if target_pct is not None and pay is not None and pay > 0:
            try:
                pct = (float(v_tp) - pay) / pay * 100.0
                tp_line += f"  ≈ +{pct:.0f}% vs debit"
            except (TypeError, ValueError):
                pass
        elif target_pct is not None:
            tp_line += f"  (+{float(target_pct) * 100:.0f}% on debit aim)"
        lines.append(tp_line)
    elif candidate.get("target_profit_pct") is not None and pay is not None:
        try:
            aim = pay * (1.0 + float(candidate["target_profit_pct"]))
            lines.append(f"Exit target: {_fmt_dollar(aim)} combo mark (+{float(candidate['target_profit_pct']) * 100:.0f}% aim)")
        except (TypeError, ValueError):
            pass

    if v_stop is not None:
        stop_line = f"Exit stop:   {_fmt_dollar(v_stop)} combo mark (hard stop)"
        if pay is not None and pay > 0:
            try:
                pct = (float(v_stop) - pay) / pay * 100.0
                stop_line += f"  ≈ {pct:.0f}% vs debit"
            except (TypeError, ValueError):
                pass
        lines.append(stop_line)

    if v0 is not None and v_tp is None and v_stop is None:
        lines.append(f"Mark anchor (V0): {_fmt_dollar(v0)}")

    return lines


def _plan_contract_dte_line(plan: dict[str, Any]) -> str | None:
    """Best-effort DTE from matched legs or candidate window."""
    matched = list(plan.get("matched_legs") or [])
    if not matched:
        matched = _legs_from_combo_spec_display(plan_combo_spec(plan))
    dtes: list[float] = []
    for leg in matched:
        try:
            d = float(leg.get("dte") or 0.0)
            if d > 0:
                dtes.append(d)
        except (TypeError, ValueError):
            continue
    if dtes:
        d = min(dtes)
        return f"DTE:       {d:.1f} days (nearest expiry leg)"
    candidate = plan.get("candidate") or {}
    dte_min = candidate.get("target_dte_min")
    dte_max = candidate.get("target_dte_max")
    if dte_min is not None and dte_max is not None:
        return f"DTE window: {dte_min}–{dte_max} days (engine target)"
    return None


def _build_robinhood_universe_message(plan: dict[str, Any]) -> str:
    """Actionable manual-buy alert when a *symbol* first becomes active in the universe."""
    sym = str(plan.get("symbol") or "?")
    pid = str(plan.get("plan_id") or "?")
    decis = plan.get("decision") or {}
    regime_key = str(plan.get("regime_key") or decis.get("regime_key") or "")
    rationale = str(decis.get("rationale") or "")
    strat_human, structure, combo_qty = _plan_option_structure_lines(plan, None)
    candidate = plan.get("candidate") or {}
    target_pct = candidate.get("target_profit_pct")
    max_risk_pct = candidate.get("max_risk_pct")
    try:
        rs = float(plan.get("rank_score") or 0.0)
        rs_fmt = f"{rs:.4f}"
    except (TypeError, ValueError):
        rs_fmt = str(plan.get("rank_score") or "?")

    lines = [
        f"🟢 ROBINHOOD — BUY IDEA — {sym}",
        ACCOUNT_ROBINHOOD,
        _SEP,
        f"Id:        {pid}",
        f"Strategy:  {strat_human}",
        f"Structure: {structure}",
        f"Combo qty: ×{combo_qty}",
        f"Rank:      {rs_fmt}",
    ]
    dte_line = _plan_contract_dte_line(plan)
    if dte_line:
        lines.append(dte_line)
    if regime_key:
        lines.append(f"Regime:    {_regime_human(regime_key)}")
    if rationale and rationale not in (strat_human, structure):
        lines.append(f"Logic:     {rationale}")
    if target_pct is not None:
        lines.append(f"Profit aim: +{float(target_pct) * 100:.0f}% on debit (model)")
    if max_risk_pct is not None:
        lines.append(f"Risk cap:   {float(max_risk_pct) * 100:.1f}% of book")
    edge = plan.get("options_edge") if isinstance(plan.get("options_edge"), dict) else {}
    if edge:
        be = edge.get("buyer_edge_pct")
        if be is not None:
            try:
                lines.append(f"Model edge: {float(be) * 100:+.1f}% vs BS fair (buyer-favorable if +)")
            except (TypeError, ValueError):
                pass
        for label, key in (("Net Δ", "net_delta"), ("Γ", "avg_gamma"), ("Θ/day", "avg_theta")):
            v = edge.get(key)
            if v is not None:
                try:
                    lines.append(f"{label:8} {float(v):+.3f}")
                except (TypeError, ValueError):
                    pass
    pay_exit = _plan_pay_and_exit_lines(plan)
    if pay_exit:
        lines.append(_SEP)
        lines.extend(pay_exit)
    lines.extend(
        [
            _SEP,
            "Paper book: same plan_id opened in local trade_log (monitor uses these marks).",
            "Session:   ideas after 09:45 ET (15m post-open); no scans outside 09:30–16:00 ET.",
            "▶ Open Robinhood and match the structure/debit above; paper book tracks TP/stop.",
        ]
    )
    return "\n".join(lines)


def _build_challenge_entry_message(pos: dict[str, Any], state: dict[str, Any]) -> str:
    bal = state.get("balance")
    lines = [
        f"🟢 RLM CHALLENGE — NEW POSITION — {pos.get('symbol', '?')}",
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
        f"✅ RLM CHALLENGE — CLOSED — {trade.get('symbol', '?')}",
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


def _build_tp_opt_message(sym: str, pid: str, mark: str, row: dict, plan: dict, root: Path) -> str:
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


def _positions_challenge_section(
    root: Path,
    *,
    max_positions: int,
    live_asof: str | None = None,
) -> list[str]:
    """Open RLM challenge positions + balance (``data/challenge/state.json``)."""
    ch_path = default_paths(root)["challenge_state"]
    lines: list[str] = [
        "─── RLM CHALLENGE ($1K→$100K · cash account · local paper) ───",
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
    open_val = sum(float(p.get("current_value") or 0) for p in opens)
    equity = bal + open_val
    lines.append(
        f"    Cash: {_fmt_dollar(bal)}  ·  equity {_fmt_dollar(equity)}  seed {_fmt_dollar(seed)}"
        f"  ·  {len(opens)} open contract leg(s)"
    )
    lu = str(raw.get("last_updated") or "").strip() or "?"
    mtime_s = (
        datetime.fromtimestamp(ch_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if ch_path.is_file()
        else "?"
    )
    lines.append(f"    State: last_updated={lu}  ·  state.json mtime {mtime_s}")
    if live_asof:
        lines.append(f"    Live marks as of {live_asof[:19].replace('T', ' ')} UTC (this request)")
    else:
        lines.append("    Live marks unavailable — showing last saved state.")
    if not opens:
        lines.append("    (no open challenge positions)")
        return lines
    sorted_o = sorted(opens, key=lambda x: (str(x.get("symbol") or ""), str(x.get("position_id") or "")))
    for pos in sorted_o[:max_positions]:
        pid = pos.get("position_id", "?")
        sym = str(pos.get("symbol") or "?")
        opt = pos.get("option_type", "?")
        strike = pos.get("strike", "")
        qty = pos.get("qty", "")
        entry = pos.get("entry_date", "?")
        dte_ent = pos.get("dte_at_entry", "")
        dte = pos.get("dte_remaining", dte_ent)
        upnl = pos.get("unrealised_pnl", 0)
        cost = pos.get("total_cost", "")
        prem_e = pos.get("premium_per_share", "")
        prem_c = pos.get("current_premium", prem_e)
        und_e = pos.get("underlying_entry", "")
        exp_d = _challenge_expiry_from_entry(str(entry), dte_ent)
        exp_disp = _expiry_mmddyy(str(exp_d.isoformat()) if exp_d is not None else "")
        sk = _fmt_strike_money(strike)
        ot = _option_type_call_put(str(opt))
        lines.append(f"  • {sym} {sk} {ot} - Exp. {exp_disp}")
        if und_e not in ("", None):
            try:
                from rlm.market.live_quotes import fetch_equity_quote

                q = fetch_equity_quote(sym) if live_asof else None
                if q is not None:
                    lines.append(
                        f"    {sym} last {_fmt_dollar(q.price)}  (entry underlying {_fmt_dollar(und_e)})"
                    )
            except Exception:  # noqa: BLE001
                pass
        cur_val = pos.get("current_value", "")
        try:
            cur_val_f = float(cur_val)
        except (TypeError, ValueError):
            try:
                cur_val_f = float(prem_c) * float(qty) * 100.0
            except (TypeError, ValueError):
                cur_val_f = None
        lines.append(f"    Cost - {_fmt_dollar(cost)}")
        if cur_val_f is not None:
            lines.append(f"    Current val - {_fmt_dollar(cur_val_f)}")
        lines.append(f"    Current PnL - {_fmt_dollar(upnl)}")
        lines.append(
            f"    challenge_id={pid}  opened {entry}  DTE_now={dte}  "
            f"option mark {_fmt_dollar(prem_c)}/sh (entry {_fmt_dollar(prem_e)}/sh)"
        )
    if len(opens) > max_positions:
        lines.append(f"    … {len(opens) - max_positions} more")
    return lines


def _expiry_mmddyy(expiry_raw: str) -> str:
    """Normalize expiry to ``MM.DD.YY`` (US-style) for Telegram."""
    s = str(expiry_raw).strip()
    if not s or s == "?":
        return "?"
    try:
        d = date.fromisoformat(s[:10])
    except ValueError:
        return s[:10] if len(s) >= 10 else s
    return f"{d.month:02d}.{d.day:02d}.{str(d.year)[-2:]}"


def _option_type_call_put(option_type: str) -> str:
    o = str(option_type).lower().strip()
    if o.startswith("c"):
        return "Call"
    if o.startswith("p"):
        return "Put"
    return (option_type or "Opt").title()


def _fmt_strike_money(strike: Any) -> str:
    try:
        x = float(strike)
    except (TypeError, ValueError):
        return "?"
    if abs(x - int(x)) < 1e-9:
        return f"${int(round(x)):,}"
    return f"${x:,.2f}"


def _large_options_position_lines(
    plan: dict,
    row: dict[str, str],
    pid: str,
    *,
    warn_suffix: str,
    live: Any | None = None,
) -> list[str]:
    """Large-options book: human option line(s) + cost / value / PnL (trade_log dollars)."""
    sym = str(row.get("symbol") or "?")
    legs = [x for x in (plan.get("matched_legs") or []) if isinstance(x, dict)]
    if not legs:
        legs = _legs_from_combo_spec_display(plan_combo_spec(plan))
    lines: list[str] = []
    strat_human, _, _ = _plan_option_structure_lines(plan, row)

    if len(legs) == 1:
        lg = legs[0]
        sk = _fmt_strike_money(lg.get("strike"))
        ot = _option_type_call_put(str(lg.get("option_type") or ""))
        exp = _expiry_mmddyy(str(lg.get("expiry") or ""))
        lines.append(f"  • {sym} {sk} {ot} - Exp. {exp}{warn_suffix}")
    elif len(legs) > 1:
        lines.append(f"  • {sym} ({strat_human}){warn_suffix}")
        for lg in legs:
            sk = _fmt_strike_money(lg.get("strike"))
            ot = _option_type_call_put(str(lg.get("option_type") or ""))
            exp = _expiry_mmddyy(str(lg.get("expiry") or ""))
            side = str(lg.get("side") or "").upper() or "—"
            lines.append(f"      {side} {sk} {ot} - Exp. {exp}")
    else:
        strat = _universe_row_strategy(plan)
        lines.append(f"  • {sym} ({strat or '—'}){warn_suffix}")

    indent = "    "
    try:
        cost = float(row.get("entry_debit") or "")
    except (TypeError, ValueError):
        cost = None
    try:
        cur = float(row.get("current_mark") or "")
    except (TypeError, ValueError):
        cur = None
    try:
        upnl = float(row.get("unrealized_pnl") or "")
    except (TypeError, ValueError):
        upnl = None

    if live is not None:
        if getattr(live, "underlying_last", None) is not None:
            lines.append(f"{indent}{sym} last {_fmt_dollar(live.underlying_last)}")
        if getattr(live, "current_mark", None) is not None:
            cur = float(live.current_mark)
        if getattr(live, "unrealized_pnl", None) is not None:
            upnl = float(live.unrealized_pnl)
        for leg_line in getattr(live, "leg_lines", None) or []:
            lines.append(leg_line)

    if cost is not None:
        lines.append(f"{indent}Cost - {_fmt_dollar(cost)}")
    if cur is not None:
        lines.append(f"{indent}Current val - {_fmt_dollar(cur)}")
    if upnl is not None:
        lines.append(f"{indent}Current PnL - {_fmt_dollar(upnl)}")

    sig = str(row.get("signal", "") or "")
    dte_r = str(row.get("dte", "") or "")
    lines.append(f"{indent}{pid}  ·  {sig}  ·  DTE={dte_r}")

    return lines


def build_universe_and_positions(root: Path, *, max_active: int = 12, max_positions: int = 20) -> str:
    """Positions grouped by trading account (large options, large equities, RLM challenge); then active universe."""
    from rlm.notify.position_marks import refresh_all_position_marks

    live_bundle = refresh_all_position_marks(root)
    live_asof = live_bundle.challenge_asof or live_bundle.asof_utc

    p = default_paths(root)
    plans_data = _read_plans(p["plans"]) or {}
    plan_lookup = _plan_by_pid(plans_data)
    plan_by_sym = _plan_by_symbol(plans_data)
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
    plan_snapshots = _load_trade_plan_snapshots(p["trade_plan_snapshots"])

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
            except (TypeError, ValueError):
                pass
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

            plan = _plan_with_trade_log_legs(
                _resolved_options_plan(plans_data, pid, plan_snapshots),
                row,
            )
            lines.extend(
                _large_options_position_lines(
                    plan,
                    row,
                    pid,
                    warn_suffix=warn_suffix,
                    live=live_bundle.options.get(pid),
                )
            )

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
            live_eq = live_bundle.equities.get(pid)
            mark = lr.get("current_mark", "") if lr else ""
            if live_eq is not None and live_eq.mark_price is not None:
                mark = f"{live_eq.mark_price:.2f}"
            usd_raw = lr.get("unrealized_pnl", "") if lr else ""
            upct_raw = lr.get("unrealized_pnl_pct", "") if lr else ""
            if live_eq is not None and live_eq.unrealized_pnl is not None:
                usd_raw = live_eq.unrealized_pnl
                upct_raw = live_eq.unrealized_pnl_pct if live_eq.unrealized_pnl_pct is not None else upct_raw
            sig = lr.get("signal", "") if lr else ""
            eq_plan = _resolved_equity_plan(plans_data, plan_lookup, plan_by_sym, pid, str(sym))
            thesis = _equity_display_thesis(eq_plan, d, lr)
            if len(thesis) > 36:
                thesis = thesis[:33] + "…"
            reg_h = _equity_display_regime(eq_plan, d)
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
    lines.extend(_positions_challenge_section(root, max_positions=max_positions, live_asof=live_asof))

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


def _active_symbols_from_plans(data: dict[str, Any]) -> set[str]:
    """Active underlying symbols (Robinhood BUY IDEA de-dupe across plan_id rotation)."""
    return _active_symbols_from_plans_payload(data)


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
        lines.append(f"  • {sym}  strategy={st}  regime={reg_h}  score={rs_fmt}  conf={conf_fmt}  id={pid}")
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


def _fmt_pnl(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.2f}"


def _challenge_pnl_section(root: Path) -> str:
    """Build the RLM Challenge section of the PnL report."""
    p = default_paths(root)
    state_path = p["challenge_state"]
    if not state_path.is_file():
        return "--- RLM Challenge ($1K → $100K · cash account · paper) ---\n  (no challenge state — run `rlm challenge --reset`)"

    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "--- RLM Challenge ($1K → $100K · cash account · paper) ---\n  (unreadable state file)"

    balance = float(raw.get("balance", 0))
    seed = float(raw.get("seed", 1000))
    target = float(raw.get("target", 100_000))
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
        "--- RLM Challenge ($1K → $100K · cash account · paper) ---",
        f"Balance:   ${balance:,.2f}",
        f"Seed:      ${seed:,.2f}",
        f"Progress:  {progress:.1f}% ({milestone_label})",
        f"Today (realized, exit date):     {_fmt_pnl(daily)}",
        f"This week (realized, exit date): {_fmt_pnl(weekly)}",
        f"Net vs seed (cash):              {_fmt_pnl(all_time)}",
        f"Open MTM (unrealized):           {_fmt_pnl(open_mtm)}",
        "Ledger:    data/processed/ledgers/rlm_challenge_book.csv",
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
        from rlm.data.ibkr_snapshot import fetch_ibkr_account_snapshot, format_account_summary_money
    except ImportError:
        return "--- IBKR ACCOUNT ---\n  (ibapi not installed)"
    try:
        snap = fetch_ibkr_account_snapshot(timeout_sec=15.0)
    except Exception:  # noqa: BLE001
        return "--- IBKR ACCOUNT ---\n  (Gateway unavailable)"

    summary = snap.account_summary
    unreal = format_account_summary_money(summary, "UnrealizedPnL")
    real = format_account_summary_money(summary, "RealizedPnL")
    if unreal == "—" and real == "—":
        note = "  (Gateway did not return PnL tags — normal on some paper accounts)"
    else:
        note = ""

    return "\n".join(
        [
            "--- IBKR ACCOUNT ---",
            f"Net liq:    {format_account_summary_money(summary, 'NetLiquidation')}",
            f"Cash:       {format_account_summary_money(summary, 'TotalCashValue')}",
            f"Buying pwr: {format_account_summary_money(summary, 'BuyingPower')}",
            f"Unreal PnL: {unreal}",
            f"Real PnL:   {real}",
            note,
        ]
    ).rstrip()


def build_pnl_text(root: Path) -> str:
    """Full P&L report across all three systems: equities, options, RLM challenge + IBKR balances."""
    ledger_note = ""
    try:
        write_trading_ledgers(root)
        ledger_note = (
            "\n--- LEDGERS (CSV for Sheets/Excel) ---\n"
            "  data/processed/ledgers/large_equities_book.csv\n"
            "  data/processed/ledgers/large_options_book.csv\n"
            "  data/processed/ledgers/rlm_challenge_book.csv\n"
        )
    except Exception as exc:  # noqa: BLE001
        ledger_note = f"\n(ledgers sync error: {exc})\n"

    sections: list[str] = ["=== P&L REPORT ==="]

    eq_rows = load_equity_trade_log_rows(root)
    eq_d, eq_w, _, _ = book_pnl_aggregates(eq_rows, book="equity")
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
                f"Realized all-time (closed only): {_fmt_pnl(eq_snap.closed_realized)}",
                f"Open MTM (unrealized):           {_fmt_pnl(eq_snap.open_mtm)}",
            ]
        )
    )

    opt_rows = load_options_trade_log_rows(root)
    opt_d, opt_w, _, _ = book_pnl_aggregates(opt_rows, book="options")
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
                f"Realized all-time (closed only): {_fmt_pnl(opt_snap.closed_realized)}",
                f"Open MTM (unrealized):           {_fmt_pnl(opt_snap.open_mtm)}",
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
        from rlm.data.ibkr_snapshot import (
            IbkrPositionRow,
            account_summary_tag_float,
            fetch_ibkr_account_snapshot,
            format_account_summary_money,
        )
    except ImportError as e:
        return f"IBKR not available: {e}"
    try:
        snap = fetch_ibkr_account_snapshot(timeout_sec=25.0)
    except Exception as e:
        return f"Could not read IBKR balances: {e}\n(Confirm Gateway is up and .env has IBKR_HOST/PORT.)"

    summary = snap.account_summary

    def _tag_raw(t: str) -> str:
        val = account_summary_tag_float(summary, t)
        if val is None:
            return "—"
        return f"{val:,.2f}"

    nlv = _tag_raw("NetLiquidation")
    cash = _tag_raw("TotalCashValue")
    bp = _tag_raw("BuyingPower")
    u_pnl = format_account_summary_money(summary, "UnrealizedPnL")

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
    """Symbols already treated as active for Robinhood BUY IDEA de-dupe (survives plan_id rotation)."""
    last_universe_active_symbols: set[str] = field(default_factory=set)
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
        raw_us = d.get("last_universe_active_symbols")
        if raw_us is not None:
            s.last_universe_active_symbols = {
                str(x).strip().upper() for x in raw_us if str(x).strip()
            }
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
            "last_universe_active_symbols": sorted(self.last_universe_active_symbols),
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
    plan_by_sym = _plan_by_symbol(plans_data)

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
        st.last_universe_active_symbols = _active_symbols_from_plans(plans_data)
        ch_path = p["challenge_state"]
        if ch_path.is_file():
            try:
                ch_raw = json.loads(ch_path.read_text(encoding="utf-8"))
                st.challenge_trade_n = len(ch_raw.get("trade_history") or [])
                st.challenge_open_ids = {
                    str(o.get("position_id")) for o in (ch_raw.get("open_positions") or []) if o.get("position_id")
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
        st.last_universe_active_symbols = _active_symbols_from_plans(plans_data)
        merged = {**state_blob, **st.to_json()}
        return [], merged

    if st.notify_seeded and "last_universe_active_symbols" not in state_blob:
        # Upgrade older state: seed symbols from the current book without re-alerting.
        st.last_universe_active_symbols = _active_symbols_from_plans(plans_data)
        merged = {**state_blob, **st.to_json()}
        return [], merged

    if st.notify_seeded and "challenge_trade_n" not in state_blob:
        ch_path = p["challenge_state"]
        if ch_path.is_file():
            try:
                ch_raw = json.loads(ch_path.read_text(encoding="utf-8"))
                st.challenge_trade_n = len(ch_raw.get("trade_history") or [])
                st.challenge_open_ids = {
                    str(o.get("position_id")) for o in (ch_raw.get("open_positions") or []) if o.get("position_id")
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
            if _notify_flag("TELEGRAM_NOTIFY_UNIVERSE", default="1") and isinstance(plan, dict) and plan:
                # Robinhood manual alert + paper row share the same universe plan (seeded at pipeline).
                pass
            else:
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
                _build_exit_opt_message(sym, pid, mark, sig, row, plan) + "\n" + _options_exit_account_impact(root)
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
    cur_syms = _active_symbols_from_plans(plans_data)
    if _notify_flag("TELEGRAM_NOTIFY_UNIVERSE", default="1"):
        # Key off symbols: rescans mint fresh ``{SYM}_{YYYYMMDD_HHMM}`` plan_ids every cycle.
        for sym in sorted(cur_syms - st.last_universe_active_symbols):
            plan = plan_by_sym.get(sym)
            if isinstance(plan, dict) and plan:
                out.append(_build_robinhood_universe_message(plan))
                pid = str(plan.get("plan_id") or "").strip()
                if pid:
                    st.announced_trade_open.add(pid)
    st.last_universe_active_ids = cur_u
    st.last_universe_active_symbols = cur_syms

    prev_eq = st.last_equity_open
    if not _notify_flag("TELEGRAM_NOTIFY_EQUITY", default="1"):
        st.last_equity_open = now_open
        if _notify_flag("TELEGRAM_NOTIFY_CHALLENGE", default="0"):
            out.extend(_challenge_notification_messages(root, st))
        return out, {**state_blob, **st.to_json()}

    for pid, d in eq.items():
        pkey = str(pid)
        pdat = d or {}
        st_eq = str(pdat.get("status") or "")
        if st_eq == "open":
            if pkey not in prev_eq and pkey not in st.announced_equity_close:
                eq_plan = _resolved_equity_plan(
                    plans_data,
                    plan_lookup,
                    plan_by_sym,
                    pkey,
                    str(pdat.get("symbol") or ""),
                )
                out.append(_build_new_equity_message(root, pkey, pdat, eq_plan))
        elif st_eq == "closed" and pkey in prev_eq and pkey not in st.announced_equity_close:
            st.announced_equity_close.add(pkey)
            ex = pdat.get("exit_reason") or pdat.get("note") or "—"
            out.append(_build_exit_equity_message(root, pkey, pdat, str(ex)))

    st.last_equity_open = now_open
    if _notify_flag("TELEGRAM_NOTIFY_CHALLENGE", default="0"):
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
