#!/usr/bin/env python3
"""
Optional offline advisory — NOT used by ROEE or Hermes crew.

Hermes \"Spock\" runs inside regime-locus-crew with tools + research_analyst SKILL.md.
This script is for manual / hook experiments: feed it an options/regime JSON blob,
get structured GO/HOLD/ABORT JSON via local Ollama.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv("/opt/enterprise/config/.env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
SPOCK_MODEL = os.getenv("SPOCK_MODEL", "qwen2.5:7b")
SPOCK_TIMEOUT = int(os.getenv("SPOCK_TIMEOUT_SECONDS", "90"))
MIN_CONF = float(os.getenv("SPOCK_MIN_CONFIDENCE", "0.65"))

log = logging.getLogger("rlm-spock-advisory")

SPOCK_SYSTEM = """You are the RLM research advisory aid (options / regime context only).

You do NOT execute trades. ROEE + Hermes crew own production decisions.

Respond ONLY in valid JSON. No markdown fences, no preamble.

Schema:
{
  "assessment": "one sentence",
  "factors": ["short factual bullets"],
  "action": "GO | HOLD | ABORT",
  "confidence": 0.00,
  "risk": "primary structural risk (greeks, regime shift, liquidity)"
}

Rules:
- confidence < 0.5 → ABORT or HOLD
- Without regime_key + strategy_name context → HOLD
- Prefer HOLD when data stale or incomplete"""


def spock_analyze(trade_context: dict) -> dict:
    prompt = f"Analyse this Regime Locus Matrix trade context:\n{json.dumps(trade_context, indent=2, default=str)}"
    t0 = datetime.now()
    raw = ""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": SPOCK_MODEL,
                "system": SPOCK_SYSTEM,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.05, "num_predict": 400},
            },
            timeout=SPOCK_TIMEOUT,
        )
        raw = str(r.json().get("response", "")).strip()
    except Exception as e:
        log.error("Spock error: %s", e)
        return {
            "proceed": False,
            "action": "ABORT",
            "confidence": 0.0,
            "assessment": f"LLM error: {e}",
            "factors": [],
            "risk": "System error",
            "raw": "",
        }

    elapsed = int((datetime.now() - t0).total_seconds() * 1000)
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {
            "proceed": False,
            "action": "ABORT",
            "confidence": 0.0,
            "assessment": "Parse failed",
            "factors": [],
            "risk": "Unparseable response",
            "raw": raw,
            "elapsed_ms": elapsed,
        }
    try:
        data = json.loads(match.group())
        confidence = float(data.get("confidence", 0.0))
        action = str(data.get("action", "ABORT")).upper()
        proceed = action == "GO" and confidence >= MIN_CONF
        return {
            "proceed": proceed,
            "action": action,
            "confidence": confidence,
            "assessment": data.get("assessment", ""),
            "factors": data.get("factors", []),
            "risk": data.get("risk", ""),
            "raw": raw,
            "elapsed_ms": elapsed,
        }
    except Exception as e:
        return {
            "proceed": False,
            "action": "ABORT",
            "confidence": 0.0,
            "assessment": f"Parse error: {e}",
            "factors": [],
            "risk": "Parse failure",
            "raw": raw,
            "elapsed_ms": elapsed,
        }


def run_test() -> None:
    ctx = {
        "strategy_name": "bull_put_spread",
        "symbol": "SPY",
        "regime_key": "bull|low_vol|high_liquidity|supportive",
        "forecast_bias": "neutral",
        "target_dte_days": 14,
        "notes": "Synthetic example — not live.",
    }
    print("\nRLM Spock advisory test\n" + "─" * 50)
    print(json.dumps(ctx, indent=2))
    print("\nQuerying Ollama...\n")
    v = spock_analyze(ctx)
    print("─" * 50)
    print(f"ACTION:     {v['action']}")
    print(f"CONFIDENCE: {v['confidence']:.0%}")
    print(f"PROCEED:    {'YES' if v['proceed'] else 'NO'}")
    print(f"ASSESSMENT: {v['assessment']}")
    print(f"RISK:       {v['risk']}")
    print(f"ELAPSED:    {v.get('elapsed_ms', 0)}ms")
    for f in v.get("factors", []):
        print(f"  • {f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [rlm-spock] %(message)s")
    parser = argparse.ArgumentParser(description="RLM offline Spock-style advisory (not production path)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        run_test()
    else:
        parser.print_help()
        sys.exit(2)
