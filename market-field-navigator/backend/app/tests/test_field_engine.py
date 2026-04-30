from app.field_engine import MarketFieldEngine


def test_generate_snapshot_contains_expected_sections() -> None:
    snapshot = MarketFieldEngine().generate_snapshot("SPY")
    assert snapshot.symbol == "SPY"
    assert len(snapshot.regime_zones) == 3
    assert snapshot.decision_summary.headline
    assert snapshot.recommended_action_label == "Setup forming, confirmation needed"
