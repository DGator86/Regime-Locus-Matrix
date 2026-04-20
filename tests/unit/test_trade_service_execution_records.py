from __future__ import annotations

from rlm.core.services.trade_service import TradeDecision, TradeRequest, TradeService


class _Broker:
    def submit_trade_decision(self, symbol, decision, paper):
        return {"success": True, "broker": "fake", "order_id": "123", "message": f"{symbol}:{paper}", "details": decision}


def test_trade_execution_record_normalized():
    svc = TradeService(broker=_Broker())
    req = TradeRequest(symbol="SPY", mode="paper")
    out = svc.execute_decision(req, TradeDecision("enter", "x", 0.2, False, {}))
    assert len(out) == 1
    assert out[0].success is True
    assert out[0].order_id == "123"
    assert out[0].broker == "fake"


def test_plan_mode_generates_non_broker_execution_record():
    svc = TradeService(broker=_Broker())
    req = TradeRequest(symbol="SPY", mode="plan")
    out = svc.execute_decision(req, TradeDecision("hold", "none", 0.0, False, {}))
    assert len(out) == 1
    assert out[0].success is True
    assert out[0].broker == "none"
