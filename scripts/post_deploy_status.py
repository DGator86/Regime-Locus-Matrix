#!/usr/bin/env python3
"""One-shot post-deploy readiness snapshot for VPS/local use (three tracks)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlm.trading.tracks import track_health  # noqa: E402


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
    root = Path(os.environ.get("RLM_ROOT") or _ROOT).resolve()
    env = _read_env(root / ".env")

    print("=== RLM Post-Deploy Snapshot (three tracks) ===")
    for tid, row in track_health(root, env=env).items():
        print(f"[{tid}] {json.dumps(row, default=str)}")
    print("")

    options_log = env.get("RLM_OPTIONS_TRADE_LOG_PATH", "data/processed/trade_log.csv")
    options_log_path = Path(options_log)
    if not options_log_path.is_absolute():
        options_log_path = root / options_log_path
    challenge_log = root / "data" / "challenge" / "trade_log.csv"
    equity_log = root / "data" / "processed" / "equity_trade_log.csv"

    print(f"RLM_OPTIONS_TRADE_LOG_PATH={options_log}")
    print(f"options_log_exists={options_log_path.exists()} path={options_log_path}")
    print(f"equity_log_exists={equity_log.exists()} path={equity_log}")
    print(f"challenge_log_exists={challenge_log.exists()} path={challenge_log}")
    print(f"TELEGRAM_NOTIFY_UNIVERSE={env.get('TELEGRAM_NOTIFY_UNIVERSE', '?')}")
    print(f"TELEGRAM_NOTIFY_CHALLENGE={env.get('TELEGRAM_NOTIFY_CHALLENGE', '?')}")
    print(f"RLM_SKIP_MASTER_CHALLENGE={env.get('RLM_SKIP_MASTER_CHALLENGE', '?')}")
    print(f"RLM_PIPELINE_DTE={env.get('RLM_PIPELINE_DTE_MIN', '?')}-{env.get('RLM_PIPELINE_DTE_MAX', '?')}")
    print(f"RLM_CHALLENGE_SCALP_DTE={env.get('RLM_CHALLENGE_SCALP_DTE_MIN', '?')}-{env.get('RLM_CHALLENGE_SCALP_DTE_MAX', '?')}")
    print("")

    kronos_url = (env.get("RLM_KRONOS_REMOTE_URL") or os.environ.get("RLM_KRONOS_REMOTE_URL") or "").strip()
    if kronos_url:
        try:
            r = requests.get(f"{kronos_url.rstrip('/')}/health", timeout=15)
            payload = (
                r.json()
                if r.headers.get("content-type", "").startswith("application/json")
                else {"raw": r.text[:200]}
            )
            print(f"kronos_remote_health_status={r.status_code}")
            print("kronos_remote_health_payload=" + json.dumps(payload))
        except Exception as exc:  # noqa: BLE001
            print(f"kronos_remote_health_error={exc}")
    else:
        print("kronos_remote_health_skipped=RLM_KRONOS_REMOTE_URL unset")

    if (root / "scripts" / "verify_kronos_gpu.py").is_file():
        import subprocess

        print("")
        print("=== Kronos predict_paths probe ===")
        proc = subprocess.run(
            [sys.executable, str(root / "scripts" / "verify_kronos_gpu.py")],
            cwd=str(root),
            env={**os.environ, **env, "RLM_ROOT": str(root)},
            capture_output=True,
            text=True,
            timeout=180,
        )
        print(proc.stdout or "")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        print(f"kronos_predict_probe_exit={proc.returncode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
