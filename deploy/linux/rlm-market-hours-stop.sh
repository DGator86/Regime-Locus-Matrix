#!/usr/bin/env bash
set -euo pipefail

ET_TIME="$(python3 -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/New_York')).strftime('%H:%M %Z'))")"
echo "[market-stop] Stopping NYSE-hours services at ET ${ET_TIME}"

systemctl stop rlm-forecast.timer || true
systemctl stop regime-locus-master.service || true

# Keep bot + crew online for alerts and operations by default.
# systemctl stop rlm-telegram.service || true
# systemctl stop regime-locus-crew.service || true

pkill -SIGTERM -f "run_universe_options_pipeline.py" 2>/dev/null || true
pkill -SIGTERM -f "run_forecast_pipeline.py" 2>/dev/null || true

echo "[market-stop] Done. Trading processes stopped at ${ET_TIME}"
