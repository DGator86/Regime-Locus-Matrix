"""RLM Dashboard API — comprehensive FastAPI backend for the React frontend."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RLM Dashboard API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = Path(os.environ.get("RLM_DATA_ROOT", "data"))
UI_DIST = Path(__file__).parent.parent.parent.parent / "apps" / "rlm_ui" / "dist"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIRECTION_LABELS = ["strongly_bearish", "bearish", "mild_bearish", "weak_bearish",
                    "neutral", "neutral", "weak_bullish", "mild_bullish", "bullish", "strongly_bullish"]
VOL_LABELS = list("ABCDEFGHIJ")


def _score_to_col(s_d: float) -> int:
    """Map S_D ∈ [-1,1] → direction bin 1-10."""
    return int(np.clip(round((s_d + 1) / 2 * 9) + 1, 1, 10))


def _score_to_row(s_v: float) -> int:
    """Map S_V ∈ [-1,1] → volatility bin 0-9 (A-J, A=low)."""
    return int(np.clip(round((s_v + 1) / 2 * 9), 0, 9))


def _state_code(s_d: float, s_v: float) -> str:
    return f"{VOL_LABELS[_score_to_row(s_v)]}{_score_to_col(s_d)}"


def _regime_labels(s_d: float, s_v: float, s_l: float, s_g: float) -> dict[str, str]:
    direction = "bull" if s_d > 0.2 else ("bear" if s_d < -0.2 else "neutral")
    volatility = "hi_vol" if s_v > 0.2 else ("lo_vol" if s_v < -0.2 else "mod_vol")
    liquidity = "liquid" if s_l > 0.1 else ("illiquid" if s_l < -0.1 else "mod_liq")
    dealer = "supportive" if s_g > 0.1 else ("hostile" if s_g < -0.1 else "neutral")
    return {"direction": direction, "volatility": volatility, "liquidity": liquidity, "dealer_flow": dealer}


def _load_features(symbol: str) -> pd.DataFrame | None:
    p = DATA_ROOT / "processed" / f"forecast_features_{symbol}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _transition_risk(s_v: float, hmm_conf: float) -> str:
    if s_v > 0.5 or hmm_conf < 0.7:
        return "HIGH"
    if s_v > 0.2 or hmm_conf < 0.85:
        return "MEDIUM"
    return "LOW"


def _recommended_action(s_d: float, s_v: float, s_l: float, s_g: float, confidence: float) -> dict:
    if s_d > 0.3 and s_v < 0.3 and confidence > 0.65:
        return {"type": "ENTER", "strategy": "Bull Call Spread", "size_pct": 1.25,
                "rationale": "Strong bullish regime with supportive dealer flows and favorable volatility environment."}
    if s_d < -0.3 and s_v > 0.3:
        return {"type": "HOLD", "strategy": "Protective Put", "size_pct": 0.5,
                "rationale": "Bearish regime with elevated volatility — defensive posture warranted."}
    if abs(s_d) < 0.15 and abs(s_v) < 0.15:
        return {"type": "HOLD", "strategy": "Iron Condor", "size_pct": 0.75,
                "rationale": "Neutral compressed regime — premium collection favored."}
    return {"type": "HOLD", "strategy": "—", "size_pct": 0.0,
            "rationale": "Mixed signals — awaiting clearer regime confirmation."}


def _next_states(state_code: str, df: pd.DataFrame | None) -> list[dict]:
    if df is None or "S_D" not in df.columns:
        return []
    df = df.dropna(subset=["S_D", "S_V"])
    codes = [_state_code(row.S_D, row.S_V) for _, row in df[["S_D", "S_V"]].iterrows()]
    trans: dict[str, int] = {}
    for i in range(len(codes) - 1):
        if codes[i] == state_code:
            trans[codes[i + 1]] = trans.get(codes[i + 1], 0) + 1
    total = sum(trans.values()) or 1
    return sorted([{"code": k, "prob": round(v / total, 2)} for k, v in trans.items()],
                  key=lambda x: -x["prob"])[:4]


def _why_rlm(s_d: float, s_v: float, s_l: float, s_g: float) -> dict:
    drivers, penalties, confluences = [], [], []
    if s_g > 0.1:
        drivers.append("Dealer Flow Support")
    if s_d > 0.2:
        drivers.append("Trend Strength")
    if s_l > 0.1:
        drivers.append("Liquidity Conditions")
    if abs(s_d) > 0.1:
        drivers.append("Price Above Value" if s_d > 0 else "Price Below Value")
    if s_v > 0.3:
        penalties.append("Volatility Expansion")
    if s_d > 0.6:
        penalties.append("Overbought Conditions")
    if s_v > 0.2:
        penalties.append("Short-Term Exhaustion")
    penalties.append("S/R Overhead" if s_d > 0 else "S/R Support Lost")
    if s_d > 0.2 and s_g > 0.1:
        confluences.append("Trend + Dealer Flow")
    if s_l > 0.1 and abs(s_v) < 0.3:
        confluences.append("Liquidity + Volatility Regime")
    confluences.append("Price Structure + VP")
    return {"top_drivers": drivers[:5], "top_penalties": penalties[:4],
            "key_confluences": confluences[:4]}


def _alerts(s_v: float, s_g: float, confidence: float) -> list[dict]:
    alerts = []
    if s_v > 0.3:
        alerts.append({"level": "warning", "title": "Transition Risk Rising",
                        "body": "Volatility expansion detected"})
    if s_g > 0.2:
        alerts.append({"level": "info", "title": "Kronos Agreement",
                        "body": "High alignment with regime"})
    if confidence > 0.85:
        alerts.append({"level": "success", "title": "No Vault Trigger",
                        "body": "Position sizing unaffected"})
    return alerts


def _build_grid(df: pd.DataFrame) -> list[dict]:
    """Build 10×10 heatmap cell data with avg return per (dir_bin, vol_bin)."""
    df = df.dropna(subset=["S_D", "S_V", "close"]).copy()
    df["dir_bin"] = df["S_D"].apply(_score_to_col)
    df["vol_bin"] = df["S_V"].apply(_score_to_row)
    fwd = df["close"].pct_change().shift(-1)
    df["fwd_ret"] = fwd

    cells = []
    for d in range(1, 11):
        for v in range(10):
            mask = (df["dir_bin"] == d) & (df["vol_bin"] == v)
            sub = df[mask]["fwd_ret"].dropna()
            cells.append({
                "dir_bin": d, "vol_bin": v,
                "state_code": f"{VOL_LABELS[v]}{d}",
                "avg_return": round(float(sub.mean()), 5) if len(sub) else 0.0,
                "count": int(len(sub)),
                "win_rate": round(float((sub > 0).mean()), 3) if len(sub) else 0.5,
            })
    return cells


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/overview/{symbol}")
def get_overview(symbol: str = "SPY"):
    df = _load_features(symbol)
    if df is None:
        raise HTTPException(404, f"No data for {symbol}")

    last = df.iloc[-1]
    s_d = float(last.get("S_D", 0))
    s_v = float(last.get("S_V", 0))
    s_l = float(last.get("S_L", 0))
    s_g = float(last.get("S_G", 0))
    hmm_state = int(last.get("hmm_state", 0)) if pd.notna(last.get("hmm_state")) else 0
    hmm_conf = float(last.get("hmm_confidence", 0.5)) if pd.notna(last.get("hmm_confidence")) else 0.5
    confidence = round(hmm_conf, 3)
    state_code = _state_code(s_d, s_v)
    labels = _regime_labels(s_d, s_v, s_l, s_g)

    prev_last = df.iloc[-6] if len(df) > 5 else df.iloc[0]
    prev_code = _state_code(float(prev_last.get("S_D", 0)), float(prev_last.get("S_V", 0)))

    # Stability score — fraction of last 10 bars in same state
    recent = df.tail(10).dropna(subset=["S_D", "S_V"])
    same = sum(1 for _, r in recent.iterrows() if _state_code(r.S_D, r.S_V) == state_code)
    stability = round(same / max(len(recent), 1), 2)

    transition_type = "UPWARD SHIFT" if s_d > float(prev_last.get("S_D", 0)) else (
        "DOWNWARD SHIFT" if s_d < float(prev_last.get("S_D", 0)) else "LATERAL")

    # Markov probability — P(staying in current state) from historical transitions
    next_st = _next_states(state_code, df)
    stay_prob = next(
        (x["prob"] for x in next_st if x["code"] == state_code), 0.5
    )
    markov_prob = round(stay_prob, 2)

    return {
        "symbol": symbol,
        "timestamp": str(last.get("timestamp", "")),
        "close": round(float(last.get("close", 0)), 2),
        "scores": {"S_D": round(s_d, 4), "S_V": round(s_v, 4),
                   "S_L": round(s_l, 4), "S_G": round(s_g, 4)},
        "market_state": labels,
        "state_code": state_code,
        "hmm_state": hmm_state,
        "hmm_confidence": round(hmm_conf, 4),
        "confidence": round(confidence * 100, 1),
        "markov_prob": markov_prob,
        "transition_risk": _transition_risk(s_v, hmm_conf),
        "action": _recommended_action(s_d, s_v, s_l, s_g, confidence),
        "risk": {
            "uncertainty_pct": round(abs(s_v) * 40 + 10, 1),
            "vault_active": s_v > 0.6,
            "vp_gating": "PASS" if s_l > 0 else "HOLD",
            "environment": "TRADEABLE" if abs(s_d) > 0.1 or abs(s_g) > 0.1 else "CHOPPY",
            "drawdown_risk": "HIGH" if s_v > 0.5 else ("MEDIUM" if s_v > 0.2 else "LOW"),
        },
        "quick_stats": _quick_stats(symbol),
        "next_states": next_st,
        "alerts": _alerts(s_v, s_g, confidence),
        "why_rlm": _why_rlm(s_d, s_v, s_l, s_g),
        "recent_transitions": {
            "prev_code": prev_code,
            "curr_code": state_code,
            "transition_type": transition_type,
            "stability_score": stability,
            "bars_in_state": same,
            "early_warning": "NONE",
        },
    }


def _quick_stats(symbol: str) -> dict:
    p = DATA_ROOT / "processed" / "trade_log.csv"
    if not p.exists():
        return {"avg_return": 0, "win_rate": 0, "trades": 0, "expectancy": 0, "best_strategy": "—"}
    df = pd.read_csv(p)
    sym_df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
    closed = sym_df[sym_df.get("closed", pd.Series(dtype=int)) == 1] if "closed" in sym_df.columns else sym_df
    if closed.empty:
        return {"avg_return": 0, "win_rate": 0, "trades": 0, "expectancy": 0, "best_strategy": "—"}
    pnl_col = "unrealized_pnl_pct" if "unrealized_pnl_pct" in closed.columns else "unrealized_pnl"
    pnl = closed[pnl_col].dropna()
    wins = (pnl > 0).sum()
    losses = (pnl <= 0).sum()
    win_rate = round(float(wins / len(pnl)), 3) if len(pnl) else 0
    avg_win = float(pnl[pnl > 0].mean()) if wins else 0
    avg_loss = abs(float(pnl[pnl <= 0].mean())) if losses else 1
    expectancy = round(win_rate * avg_win - (1 - win_rate) * avg_loss, 4)
    best = "Bull Call Spread" if float(pnl.mean()) > 0 else "Protective Put"
    return {
        "avg_return": round(float(pnl.mean()), 4),
        "win_rate": win_rate,
        "trades": int(len(closed)),
        "expectancy": expectancy,
        "best_strategy": best,
    }


@app.get("/api/v1/forecast/{symbol}")
def get_forecast(symbol: str = "SPY", bars: int = Query(252, le=1500)):
    df = _load_features(symbol)
    if df is None:
        raise HTTPException(404, f"No data for {symbol}")

    want = ["timestamp", "open", "high", "low", "close", "volume",
            "S_D", "S_V", "S_L", "S_G", "hmm_state", "hmm_confidence",
            "forecast_return_lower", "forecast_return_median", "forecast_return_upper",
            "surface_atm_forward_iv"]
    cols = [c for c in want if c in df.columns]
    out = df.tail(bars)[cols].copy()
    out["timestamp"] = out["timestamp"].astype(str)

    # Add realized vol and regime color
    out["realized_vol"] = out["close"].pct_change().rolling(20).std() * (252 ** 0.5)
    out["state_code"] = out.apply(
        lambda r: _state_code(r.get("S_D", 0), r.get("S_V", 0))
        if pd.notna(r.get("S_D")) and pd.notna(r.get("S_V")) else "E5", axis=1
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    records = out.to_dict(orient="records")
    # Replace NaN with None for JSON compliance
    return [{k: (None if (isinstance(v, float) and (v != v)) else v) for k, v in row.items()} for row in records]


@app.get("/api/v1/regime-grid/{symbol}")
def get_regime_grid(symbol: str = "SPY"):
    df = _load_features(symbol)
    if df is None:
        raise HTTPException(404, f"No data for {symbol}")

    cells = _build_grid(df)

    recent = df.tail(25).dropna(subset=["S_D", "S_V"])
    trajectory = [
        {"dir_bin": _score_to_col(r.S_D), "vol_bin": _score_to_row(r.S_V),
         "state_code": _state_code(r.S_D, r.S_V), "timestamp": str(r.timestamp)}
        for _, r in recent.iterrows()
    ]

    last = df.iloc[-1]
    s_d = float(last.get("S_D", 0))
    s_v = float(last.get("S_V", 0))

    return {
        "cells": cells,
        "trajectory": trajectory,
        "current": {
            "dir_bin": _score_to_col(s_d),
            "vol_bin": _score_to_row(s_v),
            "state_code": _state_code(s_d, s_v),
        },
    }


@app.get("/api/v1/factors/{symbol}")
def get_factors(symbol: str = "SPY"):
    df = _load_features(symbol)
    if df is None:
        raise HTTPException(404, f"No data for {symbol}")

    last = df.iloc[-1]
    factor_cols = {
        "DIRECTION": ["raw_adx_direction_bias", "raw_roc_n", "raw_ma_spread_over_atr",
                      "raw_market_breadth_ratio", "raw_relative_strength_vs_index", "S_D"],
        "VOLATILITY": ["surface_atm_forward_iv", "iv_rank", "surface_skew",
                       "surface_convexity", "surface_term_slope", "S_V"],
        "LIQUIDITY": ["bid_ask_spread", "raw_volume_imbalance", "options_volume_to_oi",
                      "raw_cvd_slope", "S_L"],
        "DEALER_FLOW": ["gex", "vanna", "charm", "dealer_position_proxy", "S_G"],
    }
    result = {}
    for category, cols in factor_cols.items():
        result[category] = []
        for c in cols:
            if c in last.index and pd.notna(last[c]):
                result[category].append({"name": c, "value": round(float(last[c]), 6)})
    return result


@app.get("/api/v1/backtest/{symbol}")
def get_backtest(symbol: str = "SPY"):
    eq_path = DATA_ROOT / "processed" / f"backtest_equity_{symbol}.csv"
    wf_path = DATA_ROOT / "processed" / f"walkforward_equity_{symbol}.csv"
    tr_path = DATA_ROOT / "processed" / f"backtest_trades_{symbol}.csv"

    equity: list[dict] = []
    if eq_path.exists():
        df = pd.read_csv(eq_path)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c or "time" in c or "stamp" in c), None)
        eq_col = next((c for c in df.columns if "equity" in c or "cum" in c or "pnl" in c), None)
        if date_col and eq_col:
            equity = df[[date_col, eq_col]].rename(
                columns={date_col: "date", eq_col: "equity"}
            ).to_dict(orient="records")

    wf_equity: list[dict] = []
    if wf_path.exists():
        df2 = pd.read_csv(wf_path)
        df2.columns = [c.lower().strip() for c in df2.columns]
        date_col = next((c for c in df2.columns if "date" in c or "time" in c), None)
        eq_col = next((c for c in df2.columns if "equity" in c or "cum" in c or "pnl" in c), None)
        if date_col and eq_col:
            wf_equity = df2[[date_col, eq_col]].rename(
                columns={date_col: "date", eq_col: "equity"}
            ).to_dict(orient="records")

    trades: list[dict] = []
    if tr_path.exists():
        df3 = pd.read_csv(tr_path)
        trades = df3.tail(100).to_dict(orient="records")

    wf_sum_path = DATA_ROOT / "processed" / f"walkforward_summary_{symbol}.csv"
    stats: dict[str, Any] = {}
    if wf_sum_path.exists():
        df4 = pd.read_csv(wf_sum_path)
        if not df4.empty:
            row = df4.iloc[-1]
            def _safe_float(v: object) -> float | None:
                try:
                    f = float(v)  # type: ignore[arg-type]
                    return None if (f != f) else round(f, 4)
                except (TypeError, ValueError):
                    return None
            stats = {k: _safe_float(v) for k, v in row.items()}

    return {"equity": equity, "wf_equity": wf_equity, "trades": trades, "stats": stats}


@app.get("/api/v1/challenge")
def get_challenge():
    p = DATA_ROOT / "challenge" / "state.json"
    if not p.exists():
        return {"balance": 1000, "total_return_pct": 0, "win_rate": 0, "trades": 0}
    return json.loads(p.read_text())


@app.get("/api/v1/universe")
def get_universe():
    plans_path = DATA_ROOT / "processed" / "universe_trade_plans.json"
    if not plans_path.exists():
        return []
    data = json.loads(plans_path.read_text())
    results = data.get("results", [])
    out = []
    for res in results:
        sym = res.get("symbol", "")
        pipe = res.get("pipeline", {})
        dec = res.get("decision", {})
        s_d = pipe.get("S_D", 0) or 0
        s_v = pipe.get("S_V", 0) or 0
        out.append({
            "symbol": sym,
            "status": res.get("status", "unknown"),
            "regime_key": pipe.get("regime_key", ""),
            "state_code": _state_code(s_d, s_v),
            "S_D": round(float(s_d), 4),
            "S_V": round(float(s_v), 4),
            "S_L": round(float(pipe.get("S_L", 0) or 0), 4),
            "S_G": round(float(pipe.get("S_G", 0) or 0), 4),
            "action": dec.get("action", ""),
            "strategy": dec.get("metadata", {}).get("strategy", ""),
            "confidence": round(float(dec.get("metadata", {}).get("confidence", 0) or 0), 3),
        })
    return out


@app.get("/api/v1/trades")
def get_trades(limit: int = Query(50, le=200)):
    p = DATA_ROOT / "processed" / "trade_log.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return df.tail(limit).replace([np.inf, -np.inf], np.nan).where(
        pd.notna(df.tail(limit)), None
    ).to_dict(orient="records")


@app.get("/api/v1/ticker")
def get_ticker():
    df = _load_features("SPY")
    if df is None:
        return []
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    close = float(last.get("close", 0))
    prev_close = float(prev.get("close", close))
    chg = close - prev_close
    chg_pct = chg / prev_close * 100 if prev_close else 0

    tickers = [
        {"symbol": "SPY", "price": round(close, 2), "change": round(chg, 2),
         "change_pct": round(chg_pct, 2)},
        {"symbol": "QQQ", "price": None, "change": None, "change_pct": None},
        {"symbol": "IWM", "price": None, "change": None, "change_pct": None},
        {"symbol": "VIX", "price": None, "change": None, "change_pct": None},
        {"symbol": "DXY", "price": None, "change": None, "change_pct": None},
    ]
    return tickers


# Serve built React app if dist exists
if UI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIST), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
