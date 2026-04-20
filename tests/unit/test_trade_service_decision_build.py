from __future__ import annotations

import pandas as pd

from rlm.core.services.trade_service import TradeRequest, TradeService


def test_trade_build_decision(monkeypatch):
    class _Pipeline:
        def __init__(self, _cfg):
            pass

        def run(self, _bars, _chain):
            class _R:
                policy_df = pd.DataFrame([{"roee_action": "enter", "roee_strategy": "put_credit", "roee_size_fraction": 0.1, "vault_triggered": False}])

            return _R()

    monkeypatch.setattr("rlm.core.services.trade_service.FullRLMPipeline", _Pipeline)

    req = TradeRequest(symbol="SPY", bars_df=pd.DataFrame([{"close": 1.0}]))
    decision = TradeService().build_decision(req)
    assert decision.action == "enter"
    assert decision.strategy == "put_credit"
