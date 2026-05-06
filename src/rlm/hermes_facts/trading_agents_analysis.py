"""Hermes fact gatherer: run TradingAgents multi-agent analysis for a symbol.

Called by the ``rlm_get_trading_agents_analysis`` Hermes tool registered in
``rlm_hermes_tools/register_rlm_tools.py``.

Returns a plain dict that is JSON-serialised before being handed to the model.
If the ``trading_agents`` optional dependency is not installed, returns a
graceful degraded response so the Hermes crew can continue without it.

Response schema (all paths)
---------------------------
{
    "available":     bool,
    "symbol":        str,
    "analysis_date": str | None,   # None on error before date is known
    "action":        str | None,   # BUY | HOLD | SELL  (None on error)
    "rationale":     str | None,
    "entry_price":   float | None,
    "stop_loss":     float | None,
    "risk_level":    str | None,   # HIGH | MODERATE | LOW  (None on error)
    "confidence":    str | None,   # HIGH | MEDIUM | LOW    (None on error)
    "error":         str | None,   # populated only on failure
}
"""

from __future__ import annotations

import logging
from datetime import date as _date
from typing import Optional

log = logging.getLogger(__name__)

_ERROR_SHAPE = {
    "available": False,
    "action": None,
    "rationale": None,
    "entry_price": None,
    "stop_loss": None,
    "risk_level": None,
    "confidence": None,
    "error": None,
}


def gather_trading_agents_analysis(symbol: str, analysis_date: Optional[str] = None) -> dict:
    """Run the TradingAgents pipeline for *symbol* and return a structured dict.

    Args:
        symbol: Ticker (e.g. ``"SPY"``, ``"NVDA"``).
        analysis_date: ISO date string ``"YYYY-MM-DD"``. Defaults to today.

    Returns:
        Dict that always contains: ``available``, ``symbol``, ``analysis_date``,
        ``action``, ``rationale``, ``entry_price``, ``stop_loss``,
        ``risk_level``, ``confidence``, ``error``.
        On success: ``available=True``, ``error=None``.
        On failure: ``available=False``, data fields are ``None``.
    """
    resolved_date = analysis_date or _date.today().strftime("%Y-%m-%d")
    base = {**_ERROR_SHAPE, "symbol": symbol, "analysis_date": resolved_date}

    try:
        from rlm.trading_agents.integration import TradingAgentsAdapter
    except ImportError as exc:
        return {
            **base,
            "error": (
                "tradingagents package not installed. "
                "Run: pip install -e '.[trading_agents]'"
            ),
        }

    try:
        adapter = TradingAgentsAdapter()
        result = adapter.analyze(symbol, resolved_date)
        return {"available": True, "error": None, **result.to_dict()}
    except Exception as exc:
        log.warning("TradingAgents analysis failed for %s: %s", symbol, exc, exc_info=True)
        return {**base, "error": str(exc)}
