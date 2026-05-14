from fastapi.testclient import TestClient
from app.main import app
from app.adapters import RLMAdapter
from app.field_engine import MarketFieldEngine


def test_snapshot_core_sections() -> None:
    snapshot = MarketFieldEngine().generate_snapshot("spy")
    assert snapshot.symbol == "SPY"
    assert len(snapshot.regime_zones) == 3
    assert len(snapshot.gamma_vectors) >= 3
    assert snapshot.field_status.force_alignment >= 0
    assert snapshot.field_status.force_alignment <= 1


def test_iv_surface_grid_is_consistent() -> None:
    snapshot = MarketFieldEngine().generate_snapshot("SPY")
    expected = snapshot.iv_surface.grid_size_x * snapshot.iv_surface.grid_size_y
    assert len(snapshot.iv_surface.points) == expected


def test_sr_walls_come_from_adapter_levels() -> None:
    adapter = RLMAdapter()
    levels = adapter.get_snapshot_inputs("SPY")["levels"]
    snapshot = MarketFieldEngine(adapter=adapter).generate_snapshot("SPY")
    prices = {wall.price for wall in snapshot.sr_walls}
    assert set(levels["support"]).issubset(prices)
    assert set(levels["resistance"]).issubset(prices)


def test_action_mapping_fallback() -> None:
    class UnknownActionAdapter(RLMAdapter):
        def get_snapshot_inputs(self, symbol: str) -> dict:
            payload = super().get_snapshot_inputs(symbol)
            payload["recommended_action"] = "SOMETHING_ELSE"
            return payload

    snapshot = MarketFieldEngine(adapter=UnknownActionAdapter()).generate_snapshot("SPY")
    assert snapshot.recommended_action_label == "No clean trade"


def test_snapshot_non_empty_vectors_and_levels() -> None:
    snapshot = MarketFieldEngine().generate_snapshot("SPY")
    assert len(snapshot.gamma_vectors) > 0
    assert len(snapshot.sr_walls) > 0
    assert len(snapshot.liquidity_wells) > 0
    assert len(snapshot.price_path) > 0


def test_snapshot_endpoint_cors_preflight() -> None:
    client = TestClient(app)
    response = client.options(
        "/api/market-field/snapshot",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
