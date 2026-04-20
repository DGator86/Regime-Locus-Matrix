from __future__ import annotations

from rlm.core.services.trade_service import TradeArtifacts, TradeDecision, TradeExecutionRecord, TradeRequest, TradeService


def test_trade_artifacts_written(tmp_path):
    svc = TradeService()
    req = TradeRequest(symbol="SPY", out_dir=tmp_path, write_artifacts=True)
    arts = svc.write_outputs(
        req,
        TradeDecision("hold", "none", 0.0, False, {}),
        [TradeExecutionRecord(success=True, broker="none", order_id=None, message="ok")],
        "run1",
    )
    assert isinstance(arts, TradeArtifacts)
    assert arts.decision_path and arts.decision_path.exists()
    assert arts.execution_path and arts.execution_path.exists()
    assert arts.manifest_path and arts.manifest_path.exists()
    assert arts.manifest_path.name == "run_manifest.json"
