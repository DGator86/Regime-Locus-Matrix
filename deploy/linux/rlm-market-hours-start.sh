#!/usr/bin/env bash
set -euo pipefail

ROOT="${RLM_ROOT:-/opt/Regime-Locus-Matrix}"
PY="${RLM_PYTHON:-/opt/rlm-venv/bin/python}"
SYNC_SYMBOLS="${RLM_SYNC_SYMBOLS:-AAPL,AMZN,GOOGL,META,MSFT,NVDA,TSLA,SPY,QQQ}"
ET_TIME="$(python3 -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/New_York')).strftime('%H:%M %Z'))")"
echo "[market-start] Current Eastern time: ${ET_TIME}"
echo "[market-start] Running startup sync (bar refresh/enrichment + preopen brief)"

if [[ -x "${PY}" && -d "${ROOT}" ]]; then
  "${PY}" "${ROOT}/scripts/append_ibkr_stock_history.py" --symbols "${SYNC_SYMBOLS}" --duration "30 D" --bar-size "1 day" || true
  "${PY}" "${ROOT}/scripts/run_universe_options_pipeline.py" --out "data/processed/universe_trade_plans.json" --no-vix || true
  "${PY}" "${ROOT}/scripts/run_session_brief.py" --phase preopen --top 8 --out "data/processed/session_brief.json" || true
else
  echo "[market-start] WARN: missing ROOT/PY (${ROOT}, ${PY}); skipping startup sync"
fi

echo "[market-start] Starting NYSE-hours services"

systemctl start rlm-master-trader.service || true
systemctl start rlm-forecast.timer || true
systemctl start rlm-systems-control-telegram.service || true
systemctl start regime-locus-crew.service || true
systemctl start rlm-challenge-loop.service || true

echo "[market-start] Done. Services started at ${ET_TIME}"
