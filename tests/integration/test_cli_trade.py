from __future__ import annotations

import sys

from rlm.cli import trade
from rlm.core.services.trade_service import TradeDecision, TradeExecutionRecord, TradeResult


def test_cli_trade_invokes_service(monkeypatch, capsys):
    class _Svc:
        def run(self, req):
            assert req.backend == "csv"
            return TradeResult(
                decision=TradeDecision("hold", "none", 0.0, False, {}),
                executions=[TradeExecutionRecord(mode="plan", success=True, message="ok")],
                run_id="abc",
            )

    monkeypatch.setattr(trade, "TradeService", lambda: _Svc())
    monkeypatch.setattr(sys, "argv", ["rlm trade", "--symbol", "SPY", "--backend", "csv"])
    trade.main()
    out = capsys.readouterr().out
    assert "Trade decision" in out
