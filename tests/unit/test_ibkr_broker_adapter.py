from __future__ import annotations

from rlm.execution.brokers.ibkr_broker import IBKRBrokerAdapter


def test_ibkr_adapter_missing_combo_spec_returns_failure():
    adapter = IBKRBrokerAdapter()
    out = adapter.submit_trade_decision("SPY", {"roee_action": "enter"}, paper=True)
    assert out["success"] is False
    assert out["broker"] == "ibkr"


def test_ibkr_adapter_option_orders_blocked_by_default():
    adapter = IBKRBrokerAdapter()
    decision = {
        "roee_action": "enter",
        "quantity": 1,
        "ibkr_combo_spec": {
            "underlying": "SPY",
            "legs": [
                {"side": "long", "option_type": "call", "strike": 400.0, "expiry": "2026-06-01"},
                {"side": "short", "option_type": "call", "strike": 405.0, "expiry": "2026-06-01"},
            ],
        },
    }
    out = adapter.submit_trade_decision("SPY", decision, paper=True)
    assert out["success"] is False
    assert "equities-only" in out["message"].lower() or "disabled" in out["message"].lower()
