"""Hermes fact gatherer: run TradingAgents multi-agent analysis for a symbol.

Called by the ``rlm_get_trading_agents_analysis`` Hermes tool registered in
``rlm_hermes_tools/register_rlm_tools.py``.

Returns a plain dict that is JSON-serialised before being handed to the model.
If the ``trading_agents`` optional dependency is not installed, returns a
graceful degraded response so the Hermes crew can continue without it.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def gather_trading_agents_analysis(symbol: str, analysis_date: Optional[str] = None) -> dict:
    """Run the TradingAgents pipeline for *symbol* and return a structured dict.

    Args:
        symbol: Ticker (e.g. ``"SPY"``, ``"NVDA"``).
        analysis_date: ISO date string ``"YYYY-MM-DD"``. Defaults to today.

    Returns:
        Dict with keys: ``symbol``, ``analysis_date``, ``action``,
        ``rationale``, ``entry_price``, ``stop_loss``, ``risk_level``,
        ``confidence``, and ``available`` (bool).
        On error: ``{"available": False, "error": "...", "symbol": symbol}``.
    """
    try:
        from rlm.trading_agents.integration import TradingAgentsAdapter
    except ImportError:
        return {
            "available": False,
            "error": (
                "tradingagents package not installed. "
                "Run: pip install -e '.[trading_agents]'"
            ),
            "symbol": symbol,
        }

    try:
        adapter = TradingAgentsAdapter()
        result = adapter.analyze(symbol, analysis_date)
        return {"available": True, **result.to_dict()}
    except Exception as exc:
        log.warning("TradingAgents analysis failed for %s: %s", symbol, exc, exc_info=True)
        return {"available": False, "error": str(exc), "symbol": symbol}
