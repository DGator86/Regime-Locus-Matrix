"""Live equity and single-leg option quotes (yfinance; optional Massive for chains)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EquityQuote:
    symbol: str
    price: float
    asof_utc: str
    source: str


def live_marks_enabled() -> bool:
    """False under pytest or when ``RLM_POSITIONS_LIVE_MARKS=0``."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    flag = (os.environ.get("RLM_POSITIONS_LIVE_MARKS") or "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def fetch_equity_quote(symbol: str) -> EquityQuote | None:
    """Last / regular-market price for an equity or ETF ticker."""
    sym = str(symbol).strip().upper()
    if not sym:
        return None
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        return None

    asof = datetime.now(tz=timezone.utc).isoformat()
    try:
        t = yf.Ticker(sym)
        fi: Any = getattr(t, "fast_info", None)
        if fi is not None:
            for key in ("lastPrice", "regularMarketPrice", "postMarketPrice", "preMarketPrice"):
                val = _float_or(getattr(fi, key, None) if not isinstance(fi, dict) else fi.get(key))
                if val is not None and val > 0:
                    return EquityQuote(sym, val, asof, "yfinance")

        hist = t.history(period="1d", interval="1m", auto_adjust=True)
        if hist is not None and not hist.empty:
            close = _float_or(hist["Close"].iloc[-1])
            if close is not None and close > 0:
                return EquityQuote(sym, close, asof, "yfinance")
    except Exception:  # noqa: BLE001
        return None
    return None


def fetch_option_mid_per_share(
    underlying: str,
    *,
    strike: float,
    expiry: date | str,
    option_type: str,
) -> float | None:
    """Mid per share for one listed contract (yfinance option chain)."""
    sym = str(underlying).strip().upper()
    try:
        exp = date.fromisoformat(str(expiry)[:10])
    except ValueError:
        return None
    ot = "call" if str(option_type).lower().startswith("c") else "put"

    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        return None

    try:
        t = yf.Ticker(sym)
        expiries = list(t.options or [])
        exp_str = exp.isoformat()
        if exp_str not in expiries:
            # Nearest expiry on or after target
            candidates = sorted(expiries)
            pick = next((e for e in candidates if e >= exp_str), None)
            if pick is None and candidates:
                pick = candidates[-1]
            if pick is None:
                return None
            exp_str = pick

        chain = t.option_chain(exp_str)
        frame = chain.calls if ot == "call" else chain.puts
        if frame is None or frame.empty:
            return None
        sub = frame.loc[(frame["strike"] - float(strike)).abs() < 0.01]
        if sub.empty:
            return None
        row = sub.iloc[0]
        bid = _float_or(row.get("bid"))
        ask = _float_or(row.get("ask"))
        last = _float_or(row.get("lastPrice"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        if last is not None and last > 0:
            return last
    except Exception:  # noqa: BLE001
        return None
    return None


def fetch_equity_quotes(symbols: list[str]) -> dict[str, EquityQuote]:
    out: dict[str, EquityQuote] = {}
    for sym in sorted({str(s).upper() for s in symbols if str(s).strip()}):
        q = fetch_equity_quote(sym)
        if q is not None:
            out[sym] = q
    return out


def _float_or(val: Any) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        f = float(val)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None
