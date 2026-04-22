# AGENTS.md

## Cursor Cloud specific instructions

### Overview

RLM (Regime Locus Matrix) is a pure-Python quantitative options-trading framework. No web server, no database, no Docker. All storage is local Parquet/CSV under `data/`.

### Quick reference

| Action | Command |
|--------|---------|
| Install (dev) | `pip install -e ".[dev]"` |
| Tests | `pytest tests/` |
| Lint (ruff) | `ruff check src/ tests/` |
| Format check | `black --check --target-version py312 src/ tests/` |
| Type check | `mypy src/` |
| Demo dataset | `python3 scripts/build_rolling_backtest_dataset.py --demo` |
| Build features | `python3 scripts/build_features.py --no-vix` |
| Forecast | `python3 scripts/run_forecast_pipeline.py --no-vix` |
| Backtest | `python3 scripts/run_backtest.py --no-vix` |
| Walk-forward | `python3 scripts/run_walkforward.py` |
| HMM forecast | `python3 scripts/run_forecast_pipeline.py --use-hmm --hmm-states 6 --no-vix` |
| Forecast (no Kronos) | `python3 scripts/run_forecast_pipeline.py --no-kronos --no-vix` |
| Backtest (no Kronos) | `python3 scripts/run_backtest.py --no-kronos --no-vix` |
| Fine-tune Kronos | `python3 scripts/finetune_kronos.py --symbol SPY --epochs 10` |
| Control Center (Streamlit) | From **repo root**: `pip install -e ".[ui]"` then `python3 scripts/run_control_center.py` (localhost). On a VPS open the app with `python3 scripts/run_control_center.py --public --port 8501` and browse to `http://<host>:8501/` (see [.streamlit/config.toml](.streamlit/config.toml); example unit: [deploy/rlm-control-center.service.example](deploy/rlm-control-center.service.example)). First tab **AT A GLANCE** summarizes universe, positions, IBKR snapshot, Telegram state, and artifact mtimes. |
| Master + Telegram (one process) | `python3 scripts/run_master.py --telegram-bot` — embeds the Telegram process; do not also run `scripts/rlm_telegram_bot.py`. Systemd: [deploy/rlm-master-telegram.service.example](deploy/rlm-master-telegram.service.example). Steps: [deploy/QUICK-ACTIVATE.txt](deploy/QUICK-ACTIVATE.txt). Telegram-only setup: [deploy/TELEGRAM_SETUP.txt](deploy/TELEGRAM_SETUP.txt). |
| Ship changes to Hostinger VPS | From repo root (Windows): `.\scripts\deploy_vps.ps1` after committing — runs `git push origin main`, SSH `git pull` on `/opt/Regime-Locus-Matrix`, restarts `rlm-telegram.service`. If the server has local edits, use `-StashOnVpsBeforePull`. Defaults: `root@2.24.28.77`; override with env `VPS_HOST`, `VPS_USER`, `VPS_REPO`. |

### Agent note: deploy after edits

When you change this codebase in a session, **commit** the intended files, then **run `.\scripts\deploy_vps.ps1`** from the repo root so the VPS matches `main` (unless the user opts out). The script requires a **clean** working tree before push.

### Gotchas

- Use `python3` (not `python`); the VM has Python 3.12 but no `python` symlink.
- `pip install` puts scripts in `~/.local/bin`; ensure `PATH` includes it (`export PATH="$HOME/.local/bin:$PATH"`).
- `--no-vix` skips yfinance VIX/VVIX download; use it for offline/isolated runs. `run_walkforward.py` does not accept `--no-vix`.
- `black --check` needs `--target-version py312` on Python 3.12 to avoid AST parse warnings for newer syntax.
- Pre-existing lint issues (133 ruff errors, 34 black reformats) exist in the repo; they are not regressions.
- External services (IBKR, Massive API, Massive S3) are optional; all unit tests use mocks/synthetic data. Set keys in `.env` (see `.env.example`) only when testing live data flows.
- `pyproject.toml` defines all config: pytest paths, ruff/black/mypy settings, and optional dependency groups (`ibkr`, `datalake`, `flatfiles`).
- Kronos (torch) is a **core** dependency (~2GB). The default model (`NeoQuasar/Kronos-mini`, 4.1M params) runs on CPU. First run downloads weights from HuggingFace. Pass `--no-kronos` to forecast/backtest scripts to skip Kronos entirely. Kronos config lives in `configs/default.yaml` under the `kronos:` block.
