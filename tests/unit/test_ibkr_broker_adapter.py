from __future__ import annotations

from rlm.execution.brokers.ibkr_broker import IBKRBrokerAdapter


def test_ibkr_adapter_missing_combo_spec_returns_failure():
    adapter = IBKRBrokerAdapter()
    out = adapter.submit_trade_decision("SPY", {"roee_action": "enter"}, paper=True)
    assert out["success"] is False
    assert out["broker"] == "ibkr"
