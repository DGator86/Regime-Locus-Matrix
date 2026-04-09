"""Event-risk detection for the ROEE pipeline.

Checks two sources in order:
1. **Earnings** — yfinance ``Ticker.calendar`` for the symbol.
2. **Macro dates** — a user-editable JSON file (``data/processed/macro_dates.json``)
   listing FOMC, CPI, triple-witching, and other systemic risk dates.

Falls back to ``False`` silently on any network or parse error so the pipeline
is never blocked by a missing API key or offline yfinance.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

try:
    import yfinance as _yf  # type: ignore[import-untyped]
    _YFINANCE_OK = True
except ImportError:
    _yf = None  # type: ignore[assignment]
    _YFINANCE_OK = False

# ---------------------------------------------------------------------------
# Macro dates file
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MACRO_DATES_PATH = _REPO_ROOT / "data" / "processed" / "macro_dates.json"

# Seed list of 2025-2026 known macro dates (FOMC, CPI, triple-witching).
# Users can edit data/processed/macro_dates.json to add/remove dates.
_SEED_MACRO_DATES: list[str] = [
    # FOMC 2025
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-11-07",
    "2025-12-17",
    # FOMC 2026
    "2026-01-28",
    "2026-03-18",
    "2026-05-06",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-11-04",
    "2026-12-16",
    # Triple-witching 2025 (3rd Friday of Mar/Jun/Sep/Dec)
    "2025-03-21",
    "2025-06-20",
    "2025-09-19",
    "2025-12-19",
    # Triple-witching 2026
    "2026-03-20",
    "2026-06-19",
    "2026-09-18",
    "2026-12-18",
    # US CPI releases 2025 (approximate; verify at bls.gov)
    "2025-01-15",
    "2025-02-12",
    "2025-03-12",
    "2025-04-10",
    "2025-05-13",
    "2025-06-11",
    "2025-07-11",
    "2025-08-12",
    "2025-09-10",
    "2025-10-09",
    "2025-11-13",
    "2025-12-11",
    # US CPI releases 2026 (approximate)
    "2026-01-14",
    "2026-02-11",
    "2026-03-11",
    "2026-04-10",
    "2026-05-13",
    "2026-06-10",
    "2026-07-15",
    "2026-08-12",
    "2026-09-11",
    "2026-10-14",
    "2026-11-12",
    "2026-12-11",
]


def _load_macro_dates(path: Path | None = None) -> set[date]:
    """Load macro risk dates from JSON, seeding the file if absent."""
    fpath = path or _DEFAULT_MACRO_DATES_PATH
    if not fpath.is_file():
        # Write seed file so users can see and edit the list.
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "description": (
                    "User-editable list of macro event dates (FOMC, CPI, triple-witching, etc.). "
                    "Add ISO-8601 date strings to suppress ROEE entries on those dates."
                ),
                "dates": sorted(_SEED_MACRO_DATES),
            }
            fpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
        raw_dates: list[str] = _SEED_MACRO_DATES
    else:
        try:
            obj = json.loads(fpath.read_text(encoding="utf-8"))
            raw_dates = list(obj.get("dates") or obj) if isinstance(obj, (dict, list)) else []
        except (json.JSONDecodeError, OSError):
            raw_dates = _SEED_MACRO_DATES

    out: set[date] = set()
    for d in raw_dates:
        try:
            out.add(date.fromisoformat(str(d).strip()))
        except ValueError:
            pass
    return out


def _is_macro_event(check_date: date, macro_dates: Iterable[date]) -> bool:
    return check_date in set(macro_dates)


# ---------------------------------------------------------------------------
# Earnings check via yfinance
# ---------------------------------------------------------------------------

# ETFs, index funds, and similar instruments that never report earnings.
# Avoids spurious yfinance HTTP 404 errors for these tickers.
_NO_EARNINGS_TICKERS: frozenset[str] = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "HYG", "LQD",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "VTI", "VOO", "VNQ", "EEM", "EFA", "VXX", "UVXY", "SOXS", "SOXL",
})


def _earnings_within_days(symbol: str, lookahead_days: int) -> bool:
    """Return True if yfinance reports an earnings date within ``lookahead_days``."""
    # Skip known ETFs / index funds — they have no earnings calendar and yfinance
    # returns HTTP 404 for them, printing noise to stderr.
    if str(symbol).upper() in _NO_EARNINGS_TICKERS:
        return False
    try:
        if not _YFINANCE_OK or _yf is None:
            return False
        ticker = _yf.Ticker(symbol)
        # yfinance sometimes prints HTTP errors to stderr; swallow them.
        with contextlib.redirect_stderr(io.StringIO()):
            cal = ticker.calendar
        if cal is None:
            return False

        today = date.today()
        cutoff = today + timedelta(days=max(0, lookahead_days))

        # yfinance calendar structure varies by version; handle dict + DataFrame.
        if hasattr(cal, "to_dict"):
            cal = cal.to_dict()

        if isinstance(cal, dict):
            for key in ("Earnings Date", "earningsDate"):
                val = cal.get(key)
                if val is None:
                    continue
                # Can be a list of Timestamps or a single value
                dates = val if isinstance(val, (list, tuple)) else [val]
                for d in dates:
                    try:
                        ed = d.date() if hasattr(d, "date") else date.fromisoformat(str(d)[:10])
                        if today <= ed <= cutoff:
                            return True
                    except (AttributeError, ValueError):
                        pass
    except Exception:  # noqa: BLE001 — yfinance errors must never block the pipeline
        pass
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def has_major_event_today(
    symbol: str,
    *,
    lookahead_days: int = 1,
    macro_dates_path: Path | None = None,
    check_earnings: bool = True,
    _today_override: date | None = None,
) -> bool:
    """Return True if ``symbol`` or the macro calendar has a major event risk today
    or within ``lookahead_days``.

    Sources checked (in order):
    1. Earnings: ``yfinance.Ticker(symbol).calendar`` within ``lookahead_days``
    2. Macro dates: ``data/processed/macro_dates.json`` (auto-seeded on first run)

    Errors in either source are swallowed; returns ``False`` on failure so the
    pipeline is never blocked by a transient network issue.
    """
    today = _today_override if _today_override is not None else date.today()

    # 1. Earnings check
    if check_earnings and _earnings_within_days(symbol, lookahead_days):
        return True

    # 2. Macro calendar check
    try:
        macro_dates = _load_macro_dates(macro_dates_path)
        cutoff = today + timedelta(days=max(0, lookahead_days))
        check = today
        while check <= cutoff:
            if _is_macro_event(check, macro_dates):
                return True
            check += timedelta(days=1)
    except Exception:  # noqa: BLE001
        pass

    return False


def reload_macro_dates(path: Path | None = None) -> set[date]:
    """Force-reload macro dates (useful after editing macro_dates.json at runtime)."""
    return _load_macro_dates(path)
