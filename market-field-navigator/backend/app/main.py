from fastapi import FastAPI, Query

from app.field_engine import MarketFieldEngine

app = FastAPI(title="Market Field Navigator Backend")
engine = MarketFieldEngine()


@app.get("/api/market-field/snapshot")
def market_field_snapshot(symbol: str = Query(default="SPY")):
    return engine.generate_snapshot(symbol)
