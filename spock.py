#!/usr/bin/env python3
"""SPOCK — Analytical Advisor for Regime Locus Matrix"""

import os, sys, json, re, logging, requests, argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("/opt/enterprise/config/.env")

OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
SPOCK_MODEL    = os.getenv("SPOCK_MODEL", "qwen2.5:7b")
SPOCK_TIMEOUT  = int(os.getenv("SPOCK_TIMEOUT_SECONDS", "90"))
MIN_CONF       = float(os.getenv("SPOCK_MIN_CONFIDENCE", "0.65"))

log = logging.getLogger("spock")

SPOCK_SYSTEM = """You are Spock — science officer and logical advisor to a trading system.
Pure data-driven analysis only. No speculation. No emotion.

Respond ONLY in valid JSON. No markdown, no prose, no preamble.

Format:
{
  "assessment": "one sentence probability statement",
  "factors": ["factor 1", "factor 2", "factor 3"],
  "action": "BUY | SELL | HOLD | ABORT",
  "confidence": 0.00,
  "risk": "primary risk in one sentence"
}

Rules:
- confidence < 0.5 → ABORT or HOLD
- Never recommend action when data insufficient
- If gnosis_projection confidence < 0.6, reduce your confidence
- If open_positions >= 5, prefer HOLD
"""

def spock_analyze(trade_context: dict) -> dict:
    prompt = f"Analyze this trade:\n{json.dumps(trade_context, indent=2, default=str)}"
    t0 = datetime.now()
    raw = ""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": SPOCK_MODEL,
            "system": SPOCK_SYSTEM,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.05, "num_predict": 400}
        }, timeout=SPOCK_TIMEOUT)
        raw = r.json().get("response", "").strip()
    except Exception as e:
        log.error("Spock error: %s", e)
        return {"proceed": False, "action": "ABORT", "confidence": 0.0,
                "assessment": f"LLM error: {e}", "factors": [], "risk": "System error", "raw": ""}

    elapsed = int((datetime.now() - t0).total_seconds() * 1000)
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"proceed": False, "action": "ABORT", "confidence": 0.0,
                "assessment": "Parse failed", "factors": [], "risk": "Unparseable response",
                "raw": raw, "elapsed_ms": elapsed}
    try:
        data = json.loads(match.group())
        confidence = float(data.get("confidence", 0.0))
        action = str(data.get("action","ABORT")).upper()
        proceed = action in ("BUY","SELL","LONG","SHORT") and confidence >= MIN_CONF
        return {
            "proceed": proceed, "action": action, "confidence": confidence,
            "assessment": data.get("assessment",""), "factors": data.get("factors",[]),
            "risk": data.get("risk",""), "raw": raw, "elapsed_ms": elapsed
        }
    except Exception as e:
        return {"proceed": False, "action": "ABORT", "confidence": 0.0,
                "assessment": f"Parse error: {e}", "factors": [], "risk": "Parse failure",
                "raw": raw, "elapsed_ms": elapsed}

def run_test():
    ctx = {
        "strategy": "regime_locus_test",
        "market": "options",
        "direction": "BUY",
        "current_price": 2.45,
        "position_size_usd": 300,
        "open_positions": 1,
        "recent_pnl_7d": 120,
        "market_state": "trending_bullish",
        "additional_context": "IV rank 34, delta 0.38, 21 DTE"
    }
    print("\n🖖 SPOCK TEST\n" + "─"*50)
    print(json.dumps(ctx, indent=2))
    print("\n⏳ Querying Spock (15-45s on CPU)...\n")
    v = spock_analyze(ctx)
    print("─"*50)
    print(f"ACTION:     {v['action']}")
    print(f"CONFIDENCE: {v['confidence']:.0%}")
    print(f"PROCEED:    {'YES ✅' if v['proceed'] else 'NO ❌'}")
    print(f"ASSESSMENT: {v['assessment']}")
    print(f"RISK:       {v['risk']}")
    print(f"ELAPSED:    {v.get('elapsed_ms',0)}ms")
    for f in v.get("factors",[]):
        print(f"  • {f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [SPOCK] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        run_test()
