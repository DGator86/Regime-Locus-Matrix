#!/usr/bin/env bash
set -euo pipefail

ROOT="${RLM_ROOT:-/opt/Regime-Locus-Matrix}"
PY="${RLM_PYTHON:-/opt/rlm-venv/bin/python}"
INTERVAL="${RLM_CHALLENGE_INTERVAL_SEC:-300}"

# PDT dry-run challenge: SPY and QQQ only (override list with RLM_CHALLENGE_SYMBOLS=SPY,QQQ or single RLM_CHALLENGE_SYMBOL=SPY).
if [[ -n "${RLM_CHALLENGE_SYMBOLS:-}" ]]; then
  IFS=',' read -ra CHALLENGE_SYM_ARRAY <<< "${RLM_CHALLENGE_SYMBOLS}"
elif [[ -n "${RLM_CHALLENGE_SYMBOL:-}" ]]; then
  CHALLENGE_SYM_ARRAY=("${RLM_CHALLENGE_SYMBOL}")
else
  CHALLENGE_SYM_ARRAY=(SPY QQQ)
fi

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
  echo "[challenge-loop] initializing challenge state (first symbol: ${CHALLENGE_SYM_ARRAY[0]})"
  "${PY}" -m rlm challenge --reset --symbol "${CHALLENGE_SYM_ARRAY[0]}" || true
fi

echo "[challenge-loop] start symbols=${CHALLENGE_SYM_ARRAY[*]} interval=${INTERVAL}s"
while true; do
  for sym in "${CHALLENGE_SYM_ARRAY[@]}"; do
    sym_trimmed="${sym// /}"
    [[ -z "${sym_trimmed}" ]] && continue
    "${PY}" -m rlm challenge --run --symbol "${sym_trimmed}" || true
  done
  sleep "${INTERVAL}"
done
