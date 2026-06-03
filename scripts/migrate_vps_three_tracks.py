#!/usr/bin/env python3
"""Apply VPS .env for three-track mode: large equities, large options (swing), SPY day trade."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_THREE_TRACK_PROFILE: dict[str, str] = {
    "RLM_STOCK_BARS_SOURCE": "eodhd",
    "RLM_ALLOW_DAILY_PRIMARY": "1",
    "RLM_PRIMARY_BAR_SIZE": "1 day",
    "RLM_PRIMARY_DURATION": "30 D",
    "RLM_PIPELINE_ARGS": (
        "--ignore-major-events --event-lookahead-days 0 --no-vix --massive-workers 4 "
        "--market-hours-only --dte-min 7 --dte-max 21"
    ),
    "RLM_PIPELINE_SHORT_DTE": "0",
    "RLM_PIPELINE_DTE_MIN": "7",
    "RLM_PIPELINE_DTE_MAX": "21",
    "RLM_PIPELINE_MARKET_HOURS_ONLY": "1",
    "RLM_MONITOR_RTH_ONLY": "1",
    "RLM_EQUITY_RTH_ONLY": "1",
    "RLM_PIPELINE_TIMEOUT_SEC": "2700",
    "RLM_SKIP_FEATURE_CSV": "1",
    "RLM_SKIP_MASTER_CHALLENGE": "1",
    "TELEGRAM_NOTIFY_UNIVERSE": "1",
    "TELEGRAM_NOTIFY_CHALLENGE": "1",
    "TELEGRAM_NOTIFY_EQUITY": "1",
    "RLM_OPTIONS_TRADE_LOG_PATH": "data/processed/options_large_account_trade_log.csv",
    "RLM_SHORT_DTE_SCORING": "0",
    "RLM_OPTIONS_MIN_BUYER_EDGE_PCT": "0.01",
    "RLM_OPTIONS_SWING_MIN_BUYER_EDGE_PCT": "0.01",
    "RLM_OPTIONS_MAX_SPREAD_PCT_MID": "0.15",
    "RLM_CHALLENGE_SYMBOL": "SPY",
    "RLM_CHALLENGE_INTERVAL_SEC": "120",
    "RLM_CHALLENGE_SCALP_DTE_MIN": "0",
    "RLM_CHALLENGE_SCALP_DTE_MAX": "5",
}


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    fast_path = repo / "scripts" / "migrate_vps_fast_universe_env.py"
    spec = importlib.util.spec_from_file_location("migrate_fast", fast_path)
    if spec is None or spec.loader is None:
        print("error: cannot load migrate_vps_fast_universe_env.py", flush=True)
        return 2
    mod = importlib.util.module_from_spec(spec)
    sys.modules["migrate_fast"] = mod
    spec.loader.exec_module(mod)
    mod._FAST_PROFILE = dict(_THREE_TRACK_PROFILE)  # type: ignore[attr-defined]
    return int(mod.main())  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())
