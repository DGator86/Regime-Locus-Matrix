#!/usr/bin/env bash
set -euo pipefail

ET_TIME="$(python3 -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/New_York')).strftime('%H:%M %Z'))")"
echo "[market-stop] Stopping NYSE-hours services at ET ${ET_TIME}"

systemctl stop rlm-forecast.timer || true
systemctl stop rlm-master-trader.service || true
systemctl stop rlm-challenge-loop.service || true

# Keep bot + crew online for alerts and operations by default.
# systemctl stop rlm-telegram.service || true
# systemctl stop regime-locus-crew.service || true

pkill -SIGTERM -f "run_universe_options_pipeline.py" 2>/dev/null || true
pkill -SIGTERM -f "run_forecast_pipeline.py" 2>/dev/null || true

# End-of-day summary prep for next day planning.
ROOT="${RLM_ROOT:-/opt/Regime-Locus-Matrix}"
PY="${RLM_PYTHON:-/opt/rlm-venv/bin/python}"
if [[ -x "${PY}" && -d "${ROOT}" ]]; then
  "${PY}" "${ROOT}/scripts/run_session_brief.py" --phase postclose --top 8 --out "data/processed/session_brief.json" || true
fi

echo "[market-stop] Done. Trading processes stopped at ${ET_TIME}"
