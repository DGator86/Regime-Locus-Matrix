"""Rolling EODHD intraday collector (5s per symbol, 09:00–16:30 ET)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from rlm.data.eodhd_stocks import (
    eodhd_poll_interval_sec,
    fetch_intraday_1m,
    is_eodhd_poll_window_open,
    load_eodhd_api_key,
)
from rlm.data.liquidity_universe import LIQUID_TEN_STOCKS_PLUS_CORE_ETFS
from rlm.data.stock_bars_provider import export_chart_bars_csv, load_stock_1m_from_lake, merge_bars_into_lake

logger = logging.getLogger(__name__)
_EASTERN = ZoneInfo("America/New_York")


def default_poll_symbols() -> tuple[str, ...]:
    """SPY/QQQ first, then remaining universe (override with ``RLM_EODHD_SYMBOLS``)."""
    raw = (os.environ.get("RLM_EODHD_SYMBOLS") or "").strip()
    if raw:
        return tuple(s.strip().upper() for s in raw.split(",") if s.strip())
    base = tuple(LIQUID_TEN_STOCKS_PLUS_CORE_ETFS)
    rest = tuple(s for s in base if s not in ("SPY", "QQQ"))
    return ("SPY", "QQQ") + rest


def _state_path(root: Path) -> Path:
    return root / "data" / "processed" / "eodhd_collector_state.json"


def _read_state(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _incremental_from_ts(symbol: str, *, root: Path, overlap_minutes: int = 5) -> int:
    lake = load_stock_1m_from_lake(symbol, root=root, lookback_days=None)
    if lake.empty:
        start = datetime.now(tz=_EASTERN) - timedelta(days=int(os.environ.get("RLM_EODHD_BACKFILL_DAYS", "30")))
        return int(start.timestamp())
    last = pd.Timestamp(lake["timestamp"].max())
    if last.tzinfo is None:
        last = last.tz_localize(_EASTERN)
    else:
        last = last.tz_convert(_EASTERN)
    return int((last - timedelta(minutes=overlap_minutes)).timestamp())


def refresh_symbol_intraday(
    symbol: str,
    *,
    root: Path,
    api_key: str | None = None,
) -> int:
    """Fetch incremental 1m bars and merge into the lake. Returns lake row count."""
    sym = str(symbol).strip().upper()
    from_ts = _incremental_from_ts(sym, root=root)
    to_ts = int(datetime.now(tz=_EASTERN).timestamp())
    if to_ts <= from_ts:
        export_chart_bars_csv(sym, root=root)
        return len(load_stock_1m_from_lake(sym, root=root))
    fresh = fetch_intraday_1m(sym, from_ts=from_ts, to_ts=to_ts, api_key=api_key)
    if fresh.empty:
        export_chart_bars_csv(sym, root=root)
        return len(load_stock_1m_from_lake(sym, root=root))
    merged = merge_bars_into_lake(sym, fresh, root=root)
    export_chart_bars_csv(sym, root=root)
    return len(merged)


def run_eodhd_collector_loop(
    *,
    root: Path,
    symbols: tuple[str, ...] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Round-robin poll: one symbol every ``RLM_EODHD_POLL_INTERVAL_SEC`` (default 5s) in-window."""
    if not load_eodhd_api_key():
        raise RuntimeError("EODHD_API_KEY is not set")

    syms = symbols or default_poll_symbols()
    interval = float(eodhd_poll_interval_sec())
    state_path = _state_path(root)
    idx = 0

    logger.info(
        "EODHD collector: %d symbols, %.0fs slot, window 09:00-16:30 ET",
        len(syms),
        interval,
    )

    while True:
        if should_stop and should_stop():
            logger.info("EODHD collector stop requested")
            return

        if not is_eodhd_poll_window_open():
            sleep_fn(30.0)
            continue

        sym = syms[idx % len(syms)]
        idx += 1
        started = time.monotonic()
        try:
            n = refresh_symbol_intraday(sym, root=root)
            state = _read_state(state_path)
            polls = state.get("polls")
            if not isinstance(polls, dict):
                polls = {}
            polls[sym] = datetime.now(tz=_EASTERN).isoformat()
            state["polls"] = polls
            state["last_symbol"] = sym
            state["last_bar_count"] = n
            _write_state(state_path, state)
            logger.info("EODHD %s: lake rows=%d", sym, n)
        except Exception:
            logger.exception("EODHD refresh failed for %s", sym)

        elapsed = time.monotonic() - started
        sleep_fn(max(0.0, interval - elapsed))
