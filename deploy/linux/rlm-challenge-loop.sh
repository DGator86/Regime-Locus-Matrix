#!/usr/bin/env bash
set -euo pipefail

ROOT="${RLM_ROOT:-/opt/Regime-Locus-Matrix}"
PY="${RLM_PYTHON:-/opt/rlm-venv/bin/python}"
SYMBOL="${RLM_CHALLENGE_SYMBOL:-SPY}"
INTERVAL="${RLM_CHALLENGE_INTERVAL_SEC:-300}"

if [[ ! -x "${PY}" ]]; then
  echo "[challenge-loop] missing python executable: ${PY}" >&2
  exit 1
fi
if [[ ! -d "${ROOT}" ]]; then
  echo "[challenge-loop] missing repo root: ${ROOT}" >&2
  exit 1
fi

cd "${ROOT}"
if [[ ! -f "data/challenge/state.json" ]]; then
  echo "[challenge-loop] initializing challenge state for ${SYMBOL}"
  "${PY}" -m rlm challenge --reset --symbol "${SYMBOL}" || true
fi

echo "[challenge-loop] start symbol=${SYMBOL} interval=${INTERVAL}s"
while true; do
  "${PY}" -m rlm challenge --run --symbol "${SYMBOL}" || true
  sleep "${INTERVAL}"
done
