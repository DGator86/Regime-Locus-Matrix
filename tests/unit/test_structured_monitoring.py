from rlm.monitoring.structured import build_pipeline_event


def test_build_pipeline_event_has_required_fields() -> None:
    out = build_pipeline_event(
        symbol="SPY",
        bar_id="2026-01-02",
        factor_values={"S_D": 0.5, "S_V": -0.2},
        regime_state="bull|low_vol|high_liquidity|supportive",
        kronos_confidence=0.71,
        action="enter",
    )
    assert out["symbol"] == "SPY"
    assert out["bar_id"] == "2026-01-02"
    assert "timestamp_utc" in out
    assert out["factor_values"]["S_D"] == 0.5
    assert out["action"] == "enter"
