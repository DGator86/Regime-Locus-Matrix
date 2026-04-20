from __future__ import annotations

from rlm.core.services.trade_service import TradeArtifacts, TradeDecision, TradeExecutionRecord, TradeRequest, TradeResult, TradeService


def test_trade_artifacts_written(tmp_path):
    svc = TradeService()
    req = TradeRequest(symbol="SPY", out_dir=tmp_path, write_artifacts=True)
    result = TradeResult(
        decision=TradeDecision("hold", "none", 0.0, False, {}),
        executions=[TradeExecutionRecord(mode="plan", success=True, message="ok")],
        run_id="run1",
    )
    arts = svc.write_outputs(req, result)
    assert isinstance(arts, TradeArtifacts)
    assert arts.decision_json and arts.decision_json.exists()
    assert arts.execution_json and arts.execution_json.exists()
    assert arts.manifest_path and arts.manifest_path.exists()
