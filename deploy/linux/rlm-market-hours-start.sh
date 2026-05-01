#!/usr/bin/env bash
set -euo pipefail

ET_TIME="$(python3 -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/New_York')).strftime('%H:%M %Z'))")"
echo "[market-start] Current Eastern time: ${ET_TIME}"
echo "[market-start] Starting NYSE-hours services"

systemctl start regime-locus-master.service || true
systemctl start rlm-forecast.timer || true
systemctl start rlm-telegram.service || true
systemctl start regime-locus-crew.service || true

echo "[market-start] Done. Services started at ${ET_TIME}"
