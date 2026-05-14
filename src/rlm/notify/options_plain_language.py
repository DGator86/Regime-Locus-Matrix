"""Plain-language explanations for option strategies and plan rows (CLI, copy/paste, reports)."""

from __future__ import annotations

from typing import Any

# Strategy keys match ``select_trade`` / ``decision.strategy_name`` naming.
_STRATEGY_TITLE: dict[str, str] = {
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


def humanize_strategy_name(strategy_name: str) -> str:
    """Short label suitable for one-line UI (same vocabulary as Telegram /positions)."""
    if not strategy_name:
        return "Unknown"
    return _STRATEGY_TITLE.get(strategy_name, strategy_name.replace("_", " ").title())


def _expiry_pretty(expiry: str) -> str:
    exp = str(expiry or "").strip()
    if len(exp) == 8 and exp.isdigit():
        return f"{exp[4:6]}.{exp[6:8]}.{exp[:4]}"
    if len(exp) >= 10 and exp[4] == "-" and exp[7] == "-":
        return f"{exp[5:7]}.{exp[8:10]}.{exp[:4]}"
    return exp or "?"


def _option_word(option_type: str) -> str:
    o = str(option_type).lower().strip()
    if o.startswith("c"):
        return "call"
    if o.startswith("p"):
        return "put"
    return "option"


def describe_leg_line(symbol: str, leg: dict[str, Any]) -> str:
    """One readable sentence for a single leg."""
    side_raw = str(leg.get("side") or "").lower().strip()
    if side_raw in ("short", "sell", "s"):
        side = "Sell"
    else:
        side = "Buy"
    strike = leg.get("strike")
    try:
        sk = f"${float(strike):,.2f}" if strike is not None else "$?"
    except (TypeError, ValueError):
        sk = str(strike) if strike is not None else "$?"
    ow = _option_word(str(leg.get("option_type") or ""))
    exp = _expiry_pretty(str(leg.get("expiry") or ""))
    qty = leg.get("quantity") or 1
    try:
        q = int(float(qty))
    except (TypeError, ValueError):
        q = 1
    qbit = f"{q}× " if q != 1 else ""
    return f"{side} {qbit}{symbol} {sk} {ow} exp {exp}"


def describe_all_legs(symbol: str, legs: list[dict[str, Any]]) -> list[str]:
    sym = str(symbol or "?").strip() or "?"
    return [describe_leg_line(sym, lg) for lg in legs if isinstance(lg, dict)]


def strategy_what_you_want(strategy_name: str, symbol: str) -> str:
    """2–4 sentences: hypothesis and risk shape (not personalized advice)."""
    s = (strategy_name or "").lower()
    sym = str(symbol or "?").strip() or "?"

    if s in ("no_trade", "no_trade_or_micro_position") or not s:
        return "No opening trade is proposed for this row."

    single_call = s in ("long_call", "aggressive_daytrader_call")
    single_put = s in ("long_put", "aggressive_daytrader_put")
    call_spread = "call" in s and "spread" in s and "iron" not in s and "condor" not in s
    put_spread = "put" in s and "spread" in s and "iron" not in s and "condor" not in s

    if single_call:
        return (
            f"You are betting {sym} will move UP before the option expires. "
            "A long call has unlimited theoretical upside and can lose the full premium paid."
        )
    if single_put:
        return (
            f"You are betting {sym} will move DOWN before the option expires. "
            "A long put pays off if the stock drops sharply; max loss is usually the premium paid."
        )
    if call_spread or "bull_call" in s:
        return (
            f"You want {sym} to RISE. A call debit spread pays best if price moves toward "
            "or through the long strike; gains cap out at the short strike. "
            "Max loss is roughly the net debit paid (plus fees)."
        )
    if put_spread or "bear_put" in s:
        return (
            f"You want {sym} to FALL. A put debit spread pays best if price drops toward "
            "or through the long strike; gains cap out at the short strike. "
            "Max loss is roughly the net debit paid (plus fees)."
        )
    if "iron_condor" in s or "strangle" in s or "straddle" in s:
        return (
            f"This structure on {sym} is mainly about RANGE or VOLATILITY: "
            "you typically need the stock to stay inside a zone (condor) or move a lot (straddle/strangle). "
            "Check the leg list for whether you are net buyer or seller of premium."
        )
    if "calendar" in s:
        return (
            f"A calendar on {sym} usually plays time decay or a volatility difference between "
            "near and far expiries; read near vs far legs below."
        )
    return (
        f"Strategy {humanize_strategy_name(strategy_name)} on {sym}. "
        "Use the leg list to see exact strikes/expiries; verify debit/credit and max loss in your platform."
    )


def _format_threshold_lines(plan: dict[str, Any]) -> list[str]:
    th = plan.get("thresholds") or {}
    if not isinstance(th, dict):
        return []
    lines: list[str] = []
    vtp = th.get("v_take_profit")
    vst = th.get("v_hard_stop")
    if vtp is not None:
        try:
            lines.append(f"Take-profit mark level (model): {float(vtp):.4f}")
        except (TypeError, ValueError):
            lines.append(f"Take-profit mark level (model): {vtp}")
    if vst is not None:
        try:
            lines.append(f"Hard-stop mark level (model): {float(vst):.4f}")
        except (TypeError, ValueError):
            lines.append(f"Hard-stop mark level (model): {vst}")
    return lines


def explain_trade_plan(plan: dict[str, Any]) -> str:
    """Multi-line plain-English block for one universe / snapshot plan row."""
    pid = str(plan.get("plan_id") or "?")
    sym = str(plan.get("symbol") or "?")
    dec = plan.get("decision") or {}
    strat = ""
    if isinstance(dec, dict):
        strat = str(dec.get("strategy_name") or "").strip()
    if not strat:
        strat = str(plan.get("strategy") or "").strip()

    title = humanize_strategy_name(strat)
    lines: list[str] = [
        f"Plan {pid} — {sym}",
        f"Structure: {title}",
        "",
        strategy_what_you_want(strat, sym),
        "",
    ]

    legs_in = plan.get("matched_legs") or []
    legs: list[dict[str, Any]] = [x for x in legs_in if isinstance(x, dict)]
    if legs:
        lines.append("Legs (plain):")
        for leg_line in describe_all_legs(sym, legs):
            lines.append(f"  • {leg_line}")
        lines.append("")

    ed = plan.get("entry_debit_dollars")
    if ed is not None:
        try:
            lines.append(f"Entry debit (plan): ${float(ed):,.2f}")
        except (TypeError, ValueError):
            lines.append(f"Entry debit (plan): {ed}")

    thold = _format_threshold_lines(plan)
    if thold:
        lines.append("")
        lines.extend(thold)

    lines.append("")
    lines.append("Note: 'Model' levels come from the pipeline/monitor; always confirm bid/ask and your broker's P/L.")
    return "\n".join(lines).rstrip() + "\n"
