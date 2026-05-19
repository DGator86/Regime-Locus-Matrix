"""EODHD intraday stock bars (primary equities/ETF feed for RLM).

Docs: https://eodhd.com/financial-apis/intraday-historical-data-api/
Limits: https://eodhd.com/financial-apis/api-limits (intraday = 5 API calls per request).

Set ``EODHD_API_KEY`` in ``.env``. US symbols use ``{TICKER}.US`` (e.g. ``SPY.US``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

_EODHD_BASE = "https://eodhd.com/api/intraday"
_EASTERN = ZoneInfo("America/New_York")


def load_eodhd_api_key() -> str | None:
    for name in ("EODHD_API_KEY", "EOD_HISTORICAL_API_KEY", "EODHD_API_TOKEN"):
        raw = (os.environ.get(name) or "").strip()
        if raw:
            return raw
    return None


def symbol_to_eodhd(symbol: str, *, exchange_suffix: str = "US") -> str:
    """Map ``SPY`` → ``SPY.US`` for EODHD."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise ValueError("empty symbol")
    if "." in sym:
        return sym
    suffix = str(exchange_suffix or "US").strip().upper()
    return f"{sym}.{suffix}"


def eodhd_poll_window_bounds() -> tuple[time, time]:
    """NYSE poll window in US/Eastern (defaults 09:00–16:30)."""
    start_raw = (os.environ.get("RLM_EODHD_POLL_START") or "09:00").strip()
    end_raw = (os.environ.get("RLM_EODHD_POLL_END") or "16:30").strip()
    return _parse_hhmm(start_raw, time(9, 0)), _parse_hhmm(end_raw, time(16, 30))


def _parse_hhmm(raw: str, default: time) -> time:
    try:
        parts = raw.split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        return time(h, m)
    except (ValueError, IndexError):
        return default


def eodhd_poll_interval_sec() -> int:
    raw = (os.environ.get("RLM_EODHD_POLL_INTERVAL_SEC") or "5").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def is_eodhd_poll_window_open(*, now: datetime | None = None) -> bool:
    """True on weekdays when Eastern clock is within the configured poll window."""
    dt = now if now is not None else datetime.now(_EASTERN)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_EASTERN)
    else:
        dt = dt.astimezone(_EASTERN)
    if dt.weekday() >= 5:
        return False
    start_t, end_t = eodhd_poll_window_bounds()
    t = dt.time().replace(second=0, microsecond=0)
    return start_t <= t < end_t


def fetch_intraday_1m(
    symbol: str,
    *,
    from_ts: int,
    to_ts: int,
    api_key: str | None = None,
    timeout_sec: float = 60.0,
) -> pd.DataFrame:
    """Pull 1-minute bars from EODHD (UTC unix ``from`` / ``to``)."""
    key = (api_key or load_eodhd_api_key() or "").strip()
    if not key:
        raise RuntimeError("EODHD_API_KEY is not set")
    eod_sym = symbol_to_eodhd(symbol)
    qs = urllib.parse.urlencode(
        {
            "api_token": key,
            "fmt": "json",
            "interval": "1m",
            "from": int(from_ts),
            "to": int(to_ts),
        }
    )
    url = f"{_EODHD_BASE}/{urllib.parse.quote(eod_sym, safe='.')}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "Regime-Locus-Matrix/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"EODHD HTTP {exc.code}: {body}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("EODHD returned invalid JSON") from exc

    return intraday_payload_to_bars(payload)


def intraday_payload_to_bars(payload: Any) -> pd.DataFrame:
    """Normalize EODHD intraday JSON to Massive/IBKR-aligned OHLCV schema."""
    rows: list[dict[str, Any]] = payload if isinstance(payload, list) else []
    if not rows and isinstance(payload, dict):
        # Some responses wrap rows under a key
        for key in ("data", "intraday", "values"):
            block = payload.get(key)
            if isinstance(block, list):
                rows = block
                break

    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts_raw = row.get("datetime") or row.get("timestamp") or row.get("date")
        if ts_raw is None:
            continue
        try:
            ts_unix = int(float(ts_raw))
        except (TypeError, ValueError):
            ts_parsed = pd.to_datetime(ts_raw, errors="coerce", utc=True)
            if pd.isna(ts_parsed):
                continue
            ts_naive = ts_parsed.tz_convert(_EASTERN).tz_localize(None)
        else:
            ts_naive = pd.to_datetime(ts_unix, unit="s", utc=True).tz_convert(_EASTERN).tz_localize(None)

        o = float(row.get("open", float("nan")))
        h = float(row.get("high", float("nan")))
        low_v = float(row.get("low", float("nan")))
        c = float(row.get("close", float("nan")))
        vol = float(row.get("volume", 0) or 0)
        vwap = (o + h + low_v + c) / 4.0 if all(pd.notna(x) for x in (o, h, low_v, c)) else c
        records.append(
            {
                "timestamp": ts_naive,
                "open": o,
                "high": h,
                "low": low_v,
                "close": c,
                "volume": vol,
                "vwap": vwap,
            }
        )

    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])

    out = pd.DataFrame.from_records(records)
    return out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def fetch_intraday_lookback_days(
    symbol: str,
    *,
    lookback_days: int = 30,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Backfill up to ``lookback_days`` of 1m bars (EODHD max 120d for 1m)."""
    end = datetime.now(tz=_EASTERN)
    start = end - pd.Timedelta(days=max(1, int(lookback_days)))
    return fetch_intraday_1m(
        symbol,
        from_ts=int(start.timestamp()),
        to_ts=int(end.timestamp()),
        api_key=api_key,
    )
