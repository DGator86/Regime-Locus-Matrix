#!/usr/bin/env bash
# Unified stack: master loop + equity paper + embedded Telegram (via --telegram-bot on run_everything).
# From repo root:
#   bash scripts/activate_rlm_stack.sh
#   bash scripts/activate_rlm_stack.sh --run
#
# Do not run rlm_telegram_bot.py separately if you use --run here (duplicates the bot).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [[ ! -f pyproject.toml ]]; then
  echo "error: no pyproject.toml in $REPO" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "warning: no .env — copy from .env.example" >&2
fi
if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
elif [[ -n "${RLM_VENV:-}" && -x "${RLM_VENV}/bin/python" ]]; then
  PY="${RLM_VENV}/bin/python"
elif command -v python3 >/dev/null; then
  PY="$(command -v python3)"
else
  echo "error: set RLM_VENV to your venv or create .venv" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
echo "REPO=$REPO  PY=$PY"
if [[ "${1:-}" == "--run" ]]; then
  exec "$PY" "$REPO/scripts/run_master.py" --telegram-bot
else
  echo "Dry run. Start with:  bash $0 --run"
fi
