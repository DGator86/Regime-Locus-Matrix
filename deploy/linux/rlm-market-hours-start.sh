#!/usr/bin/env bash
set -euo pipefail

ROOT="${RLM_ROOT:-/opt/Regime-Locus-Matrix}"
PY="${RLM_PYTHON:-/opt/rlm-venv/bin/python}"
SYNC_SYMBOLS="${RLM_SYNC_SYMBOLS:-AAPL,AMZN,GOOGL,META,MSFT,NVDA,TSLA,AMD,AVGO,JPM,SPY,QQQ}"
ET_TIME="$(python3 -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/New_York')).strftime('%H:%M %Z'))")"
echo "[market-start] Current Eastern time: ${ET_TIME}"
echo "[market-start] Running startup sync (bar refresh/enrichment + preopen brief)"

if [[ -x "${PY}" && -d "${ROOT}" ]]; then
  if [[ -f "${ROOT}/scripts/migrate_vps_three_tracks.py" ]]; then
    echo "[market-start] sync three-track .env profile"
    "${PY}" "${ROOT}/scripts/migrate_vps_three_tracks.py" || echo "[market-start] WARN: migrate_vps_three_tracks failed (non-fatal)"
  fi
  echo "[market-start] decision-tree health (offline snapshot)"
  "${PY}" "${ROOT}/scripts/run_startup_decision_tree_health.py" || echo "[market-start] WARN: startup decision-tree health failed (non-fatal)"
  if [[ -f "${ROOT}/scripts/verify_kronos_gpu.py" ]]; then
    echo "[market-start] Kronos GPU probe"
    "${PY}" "${ROOT}/scripts/verify_kronos_gpu.py" || echo "[market-start] WARN: Kronos GPU probe failed (non-fatal)"
  fi
  echo "[market-start] refresh daily bars CSV (yfinance + 1m lake)"
  "${PY}" "${ROOT}/scripts/refresh_universe_daily_bars.py" || echo "[market-start] WARN: daily bars refresh failed (non-fatal)"
  echo "[market-start] EODHD 1m backfill (if key set) + collector"
  systemctl start rlm-eodhd-stock-collector.service || true
  "${PY}" "${ROOT}/scripts/run_eodhd_stock_collector.py" --backfill --once || true
  echo "[market-start] universe pipeline (large-options swing from RLM_PIPELINE_ARGS)"
  read -r -a _PIPE_ARGS <<< "${RLM_PIPELINE_ARGS:---ignore-major-events --event-lookahead-days 0 --no-vix --massive-workers 4 --market-hours-only --dte-min 7 --dte-max 21 --no-feature-csv}"
  _TRADE_LOG="${RLM_OPTIONS_TRADE_LOG_PATH:-data/processed/options_large_account_trade_log.csv}"
  _PIPE_HAS_TRADE_LOG=0
  for _a in "${_PIPE_ARGS[@]+"${_PIPE_ARGS[@]}"}"; do
    if [[ "${_a}" == "--trade-log" ]]; then
      _PIPE_HAS_TRADE_LOG=1
      break
    fi
  done
  _TRADE_LOG_ARGS=()
  if [[ "${_PIPE_HAS_TRADE_LOG}" -eq 0 ]]; then
    _TRADE_LOG_ARGS=(--trade-log "${_TRADE_LOG}")
  fi
  "${PY}" "${ROOT}/scripts/run_universe_options_pipeline.py" \
    --out "data/processed/universe_trade_plans.json" \
    "${_TRADE_LOG_ARGS[@]}" \
    "${_PIPE_ARGS[@]}" || true
  "${PY}" "${ROOT}/scripts/run_session_brief.py" --phase preopen --top 8 --out "data/processed/session_brief.json" || true
else
  echo "[market-start] WARN: missing ROOT/PY (${ROOT}, ${PY}); skipping startup sync"
fi

echo "[market-start] Starting NYSE-hours services"

if [[ -f "${ROOT}/scripts/rlm_enable_startup_services.sh" ]]; then
  bash "${ROOT}/scripts/rlm_enable_startup_services.sh" || echo "[market-start] WARN: rlm_enable_startup_services failed (non-fatal)"
else
  systemctl start ollama.service || true
  systemctl start rlm-host-watchdog.service || true
  systemctl start rlm-master-trader.service || true
  systemctl start rlm-forecast.timer || true
  systemctl start rlm-systems-control-telegram.service || true
  systemctl start regime-locus-crew.service || true
  systemctl start rlm-challenge-loop.service || true
fi

echo "[market-start] Done. Services started at ${ET_TIME}"
