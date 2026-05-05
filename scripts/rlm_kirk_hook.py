#!/usr/bin/env python3
"""
Optional pre-trade hook wrapper around the offline Spock advisory.

Production trading decisions come from ROEE + pipelines; Hermes \"Kirk\" lives in regime-locus-crew.
This module is for experiments or custom orchestration only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/opt/enterprise/config/.env")

_HERE = Path(__file__).resolve().parent
_ENTERPRISE_AGENTS = Path("/opt/enterprise/agents")
for _d in (_HERE, _ENTERPRISE_AGENTS):
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

try:
    from spock import spock_analyze
except ImportError:
    from rlm_spock_advisory import spock_analyze

log = logging.getLogger("rlm-kirk-hook")

SPOCK_ENABLED = os.getenv("SPOCK_ENABLED", "true").lower() == "true"
OVERRIDE_KEY = os.getenv("SPOCK_OVERRIDE_KEY", "")
DECISION_LOG = os.getenv("DECISION_LOG", "/opt/enterprise/data/decisions.jsonl")


@dataclass
class KirkDecision:
    proceed: bool
    action: str
    confidence: float
    reason: str
    risk: str
    factors: list = field(default_factory=list)
    elapsed_ms: int = 0
    bypassed: bool = False


def consult_spock(trade_context: dict, override: str = "") -> KirkDecision:
    if override and override == OVERRIDE_KEY:
        return KirkDecision(True, "MANUAL_OVERRIDE", 1.0, "Override", "None", bypassed=True)
    if not SPOCK_ENABLED:
        return KirkDecision(True, "UNANALYZED", 0.5, "Advisory disabled", "None", bypassed=True)
    v = spock_analyze(trade_context)
    d = KirkDecision(
        proceed=v["proceed"],
        action=v["action"],
        confidence=v["confidence"],
        reason=v["assessment"],
        risk=v["risk"],
        factors=v.get("factors", []),
        elapsed_ms=v.get("elapsed_ms", 0),
    )
    _log_decision(trade_context, d)
    if d.proceed:
        log.info("ADVISORY GO %s @ %.0f%% (%dms)", d.action, d.confidence * 100, d.elapsed_ms)
    else:
        log.info("ADVISORY HOLD/ABORT %s @ %.0f%% — %s", d.action, d.confidence * 100, d.risk)
    return d


def _log_decision(ctx: dict, d: KirkDecision) -> None:
    try:
        Path(DECISION_LOG).parent.mkdir(parents=True, exist_ok=True)
        with open(DECISION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "context": ctx, "decision": vars(d)}) + "\n")
    except Exception as e:
        log.error("Decision log error: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [rlm-kirk-hook] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    if args.stats:
        try:
            lines = open(DECISION_LOG, encoding="utf-8").readlines()
            total = len(lines)
            approved = sum(1 for line in lines if '"proceed": true' in line)
            print(f"\nDecisions: {total} | Approved: {approved} | Vetoed: {total - approved}")
        except FileNotFoundError:
            print("No decisions logged yet.")
