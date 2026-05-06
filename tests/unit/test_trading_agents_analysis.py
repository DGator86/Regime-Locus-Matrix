from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import rlm.hermes_facts.trading_agents_analysis as ta_module


def test_returns_unavailable_when_package_not_installed(monkeypatch):
    """Graceful degradation: missing tradingagents package → available=False."""
    # Simulate ImportError inside the fact gatherer
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def fake_import(name, *args, **kwargs):
        if name == "rlm.trading_agents.integration":
            raise ImportError("tradingagents not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(ta_module, "gather_trading_agents_analysis", ta_module.gather_trading_agents_analysis)

    # Patch at the module level: make the import inside the function fail
    import rlm.trading_agents.integration as _integration_mod  # noqa: F401 — ensure it's cached

    # Replace TradingAgentsAdapter with one that raises ImportError
    original_adapter = getattr(_integration_mod, "TradingAgentsAdapter", None)

    class _BrokenAdapter:
        def __init__(self, *a, **kw):
            raise ImportError("tradingagents not installed")

    monkeypatch.setattr(_integration_mod, "TradingAgentsAdapter", _BrokenAdapter)

    result = ta_module.gather_trading_agents_analysis("SPY")

    assert result["available"] is False
    assert "error" in result
    assert result["symbol"] == "SPY"


def test_returns_unavailable_on_adapter_runtime_error(monkeypatch):
    """Any exception from the adapter propagates as available=False."""
    import rlm.trading_agents.integration as _integration_mod

    class _FlakyAdapter:
        def __init__(self, *a, **kw):
            pass

        def analyze(self, symbol, date=None):
            raise RuntimeError("upstream LLM timeout")

    monkeypatch.setattr(_integration_mod, "TradingAgentsAdapter", _FlakyAdapter)

    result = ta_module.gather_trading_agents_analysis("NVDA")

    assert result["available"] is False
    assert "upstream LLM timeout" in result["error"]
    assert result["symbol"] == "NVDA"


def test_happy_path_returns_structured_result(monkeypatch):
    """Successful analysis returns available=True with expected keys."""
    import rlm.trading_agents.integration as _integration_mod
    from rlm.trading_agents.integration import TradingAgentsResult

    mock_result = TradingAgentsResult(
        symbol="SPY",
        analysis_date="2026-05-06",
        action="BUY",
        rationale="Strong momentum confirmed by fundamentals.",
        entry_price=520.0,
        stop_loss=510.0,
        risk_level="MODERATE",
        confidence="HIGH",
    )

    class _MockAdapter:
        def __init__(self, *a, **kw):
            pass

        def analyze(self, symbol, date=None):
            return mock_result

    monkeypatch.setattr(_integration_mod, "TradingAgentsAdapter", _MockAdapter)

    result = ta_module.gather_trading_agents_analysis("SPY", "2026-05-06")

    assert result["available"] is True
    assert result["action"] == "BUY"
    assert result["symbol"] == "SPY"
    assert result["analysis_date"] == "2026-05-06"
    assert result["entry_price"] == 520.0
    assert result["stop_loss"] == 510.0
    assert result["confidence"] == "HIGH"


def test_happy_path_none_prices_allowed(monkeypatch):
    """entry_price and stop_loss may be None (LLM didn't produce them)."""
    import rlm.trading_agents.integration as _integration_mod
    from rlm.trading_agents.integration import TradingAgentsResult

    mock_result = TradingAgentsResult(
        symbol="QQQ",
        analysis_date="2026-05-06",
        action="HOLD",
        rationale="Mixed signals; no clear directional bias.",
        entry_price=None,
        stop_loss=None,
        risk_level="LOW",
        confidence="MEDIUM",
    )

    class _MockAdapter:
        def __init__(self, *a, **kw):
            pass

        def analyze(self, symbol, date=None):
            return mock_result

    monkeypatch.setattr(_integration_mod, "TradingAgentsAdapter", _MockAdapter)

    result = ta_module.gather_trading_agents_analysis("QQQ")

    assert result["available"] is True
    assert result["entry_price"] is None
    assert result["stop_loss"] is None
