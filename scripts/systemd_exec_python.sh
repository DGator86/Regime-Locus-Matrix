#!/usr/bin/env bash
# Run a repo script with the first usable Python interpreter (systemd ExecStart helper).
# Resolves, in order:
#   1. $INSTALL_ROOT/.venv/bin/python
#   2. $RLM_VENV_PYTHON (from Environment= or .env) if executable
#   3. /opt/rlm-venv/bin/python (common Hostinger layout)
#   4. python3 from PATH
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
pick_python() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    echo "$ROOT/.venv/bin/python"
  elif [[ -n "${RLM_VENV_PYTHON:-}" && -x "${RLM_VENV_PYTHON}" ]]; then
    echo "${RLM_VENV_PYTHON}"
  elif [[ -x /opt/rlm-venv/bin/python ]]; then
    echo /opt/rlm-venv/bin/python
  elif [[ -x /opt/rlm-venv/bin/python3 ]]; then
    echo /opt/rlm-venv/bin/python3
  else
    command -v python3
  fi
}
PY="$(pick_python)"
exec "$PY" "$@"
