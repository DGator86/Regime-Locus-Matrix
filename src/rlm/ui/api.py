from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from pathlib import Path
import os
from typing import Dict, List, Any

app = FastAPI(title="RLM Dashboard API")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = Path(os.environ.get("RLM_DATA_ROOT", "data"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/summary")
def get_summary():
    """Returns a high-level summary for the top metrics."""
    try:
        challenge_path = DATA_ROOT / "challenge" / "state.json"
        challenge = json.loads(challenge_path.read_text()) if challenge_path.exists() else {}
        
        trade_log_path = DATA_ROOT / "processed" / "trade_log.csv"
        trade_log = pd.read_csv(trade_log_path) if trade_log_path.exists() else pd.DataFrame()
        
        realized_pnl = 0
        unrealized_pnl = 0
        if not trade_log.empty:
            realized_pnl = float(trade_log[trade_log["closed"] == 1]["unrealized_pnl"].sum())
            unrealized_pnl = float(trade_log[trade_log["closed"] == 0]["unrealized_pnl"].sum())
            
        return {
            "balance": challenge.get("balance", 0),
            "return_pct": challenge.get("total_return_pct", 0),
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "win_rate": challenge.get("win_rate", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/matrix")
def get_matrix_data():
    """Returns the latest regime locus matrix data."""
    try:
        plans_path = DATA_ROOT / "processed" / "universe_trade_plans.json"
        if not plans_path.exists():
            return {"error": "No plans data found"}
            
        data = json.loads(plans_path.read_text())
        results = data.get("results", [])
        
        # Extract regime scores for all symbols
        matrix = []
        for res in results:
            sym = res.get("symbol")
            pipe = res.get("pipeline", {})
            matrix.append({
                "symbol": sym,
                "regime": pipe.get("regime_key"),
                "direction": pipe.get("S_D", 0),
                "volatility": pipe.get("S_V", 0),
                "liquidity": pipe.get("S_L", 0),
                "growth": pipe.get("S_G", 0),
                "confidence": res.get("decision", {}).get("metadata", {}).get("confidence", 0),
            })
            
        return matrix
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades")
def get_recent_trades():
    """Returns recent trade activity."""
    try:
        trade_log_path = DATA_ROOT / "processed" / "trade_log.csv"
        if not trade_log_path.exists():
            return []
            
        df = pd.read_csv(trade_log_path)
        # Get latest 50 rows
        recent = df.tail(50).to_dict(orient="records")
        return recent
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
