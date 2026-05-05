from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.field_engine import MarketFieldEngine
from app.schemas import MarketFieldSnapshot

app = FastAPI(title="Market Field Navigator Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = MarketFieldEngine()


@app.get(
    "/api/market-field/snapshot",
    response_model=MarketFieldSnapshot,
    summary="Market Field Snapshot",
    description=(
        "Return the current 3D market-field payload used by the frontend "
        "scene for a given symbol."
    ),
    response_description="Structured market-field snapshot for rendering and HUD display.",
)
def market_field_snapshot(
    symbol: str = Query(
        default="SPY",
        description="Ticker symbol to generate a snapshot for.",
        examples=["SPY", "QQQ"],
    )
):
    return engine.generate_snapshot(symbol)
