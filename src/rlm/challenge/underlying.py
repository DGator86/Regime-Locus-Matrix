"""Resolve session underlying for the challenge loop (live first, pipeline fallback)."""

from __future__ import annotations

import sys
from datetime import date

from rlm.market.live_quotes import fetch_equity_quote, live_marks_enabled


def resolve_session_underlying(
    symbol: str,
    pipeline_close: float | None = None,
    *,
    pipeline_bar_date: str | None = None,
    override: float | None = None,
    session_date: str | None = None,
) -> tuple[float, str]:
    """Return ``(price, source)`` for challenge entries and synthetic marks.

    Live quote is preferred whenever available. Pipeline close is used only when
    live data is unavailable (offline tests, yfinance failure).

    When neither source is available, returns ``(0.0, "unavailable")``. Callers
    must not treat that as a tradable spot — open legs should mark from
    ``underlying_entry`` instead of inventing a placeholder price.
    """
    if override is not None and override > 0:
        return float(override), "override"

    session_date = session_date or date.today().isoformat()

    if live_marks_enabled():
        quote = fetch_equity_quote(symbol)
        if quote is not None and quote.price > 0:
            if pipeline_close and pipeline_close > 0:
                drift = abs(quote.price - pipeline_close) / pipeline_close
                stale = _pipeline_is_stale(pipeline_bar_date, session_date)
                if drift > 0.02 or stale:
                    bar_note = pipeline_bar_date or "unknown date"
                    print(
                        f"[challenge] live {symbol} ${quote.price:.2f} "
                        f"vs pipeline ${pipeline_close:.2f} ({bar_note}) — using live",
                        file=sys.stderr,
                    )
            return quote.price, "live"

    if pipeline_close and pipeline_close > 0:
        if _pipeline_is_stale(pipeline_bar_date, session_date):
            print(
                f"[challenge] WARNING: pipeline {symbol} close ${pipeline_close:.2f} "
                f"from {pipeline_bar_date or '?'} is stale; live quote unavailable",
                file=sys.stderr,
            )
        return float(pipeline_close), "pipeline"

    # Never invent a spot (legacy hardcoded 500.0 marked SPY ~$700 books as a crash
    # and false-stopped calls / false-took puts). Callers must treat <=0 as unavailable.
    return 0.0, "unavailable"


def _pipeline_is_stale(pipeline_bar_date: str | None, session_date: str) -> bool:
    if not pipeline_bar_date:
        return False
    try:
        bar_day = date.fromisoformat(str(pipeline_bar_date)[:10])
        sess_day = date.fromisoformat(str(session_date)[:10])
    except ValueError:
        return False
    return (sess_day - bar_day).days > 2
