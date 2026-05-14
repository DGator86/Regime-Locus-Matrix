#!/usr/bin/env python3
"""Emit pipeline artefact snapshot JSON for ops (run on VPS)."""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path("/opt/Regime-Locus-Matrix")
PROC = ROOT / "data" / "processed"


def _age_h(path: Path) -> float | None:
    if not path.is_file():
        return None
    return (time.time() - path.stat().st_mtime) / 3600.0


def main() -> None:
    out: dict = {"root": str(ROOT), "files": {}, "live_regime_model": None, "universe_plans": None, "gate_state": None}

    for name in (
        "universe_trade_plans.json",
        "trade_log.csv",
        "equity_positions_state.json",
        "live_regime_model.json",
        "gate_state.json",
        "live_nightly_hyperparams.json",
        "regime_transition_calibration.json",
        "trade_monitor_state.json",
    ):
        p = PROC / name
        out["files"][name] = {
            "exists": p.is_file(),
            "age_hours": round(_age_h(p), 3) if p.is_file() else None,
            "size_bytes": p.stat().st_size if p.is_file() else None,
        }

    lp = PROC / "live_regime_model.json"
    if lp.is_file():
        raw = json.loads(lp.read_text(encoding="utf-8"))
        th = raw.get("timeframe_hierarchy") or {}
        out["live_regime_model"] = {
            "model": raw.get("model"),
            "primary_bar_size": th.get("primary_bar_size"),
            "primary_duration": th.get("primary_duration"),
            "confirmation_bar_sizes": th.get("confirmation_bar_sizes"),
            "top_keys": list(raw.keys())[:20],
        }

    up = PROC / "universe_trade_plans.json"
    if up.is_file():
        u = json.loads(up.read_text(encoding="utf-8"))
        rows = []
        for r in u.get("results", []):
            dec = r.get("decision") or {}
            rows.append(
                {
                    "symbol": r.get("symbol"),
                    "status": r.get("status"),
                    "action": dec.get("action"),
                    "strategy_name": dec.get("strategy_name"),
                    "skip_reason": (r.get("skip_reason") or "")[:80],
                    "regime_key": (dec.get("regime_key") or r.get("pipeline", {}).get("regime_key")),
                }
            )
        out["universe_plans"] = {"generated_at_utc": u.get("generated_at_utc"), "symbols": rows}

    gp = PROC / "gate_state.json"
    if gp.is_file():
        out["gate_state"] = json.loads(gp.read_text(encoding="utf-8"))

    ch = ROOT / "data" / "challenge" / "state.json"
    if ch.is_file():
        out["challenge_state"] = json.loads(ch.read_text(encoding="utf-8"))

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
