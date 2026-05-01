from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.field_engine import MarketFieldEngine

app = FastAPI(title="Market Field Navigator Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = MarketFieldEngine()


@app.get("/api/market-field/snapshot")
def market_field_snapshot(symbol: str = Query(default="SPY")):
    return engine.generate_snapshot(symbol)
