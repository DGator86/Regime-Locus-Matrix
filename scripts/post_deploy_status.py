#!/usr/bin/env python3
"""One-shot post-deploy readiness snapshot for VPS/local use."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests


def _read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = _read_env(root / ".env")
    options_log = env.get("RLM_OPTIONS_TRADE_LOG_PATH", "data/processed/trade_log.csv")
    options_log_path = Path(options_log)
    if not options_log_path.is_absolute():
        options_log_path = root / options_log_path
    challenge_log = root / "data" / "challenge" / "trade_log.csv"

    print("=== RLM Post-Deploy Snapshot ===")
    print(f"RLM_OPTIONS_TRADE_LOG_PATH={options_log}")
    print(f"options_log_exists={options_log_path.exists()} path={options_log_path}")
    print(f"challenge_log_exists={challenge_log.exists()} path={challenge_log}")
    print(f"telegram_balances_enabled=True (uses IBKR snapshot)")
    print("")

    runpod = "https://o69xwhi8diaukn-8000.proxy.runpod.net/health"
    try:
        r = requests.get(runpod, timeout=10)
        payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text[:200]}
        print(f"runpod_health_status={r.status_code}")
        print("runpod_health_payload=" + json.dumps(payload))
    except Exception as exc:  # noqa: BLE001
        print(f"runpod_health_error={exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
