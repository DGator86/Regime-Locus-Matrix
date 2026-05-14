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
| Walk-forward (universe batch + history) | `python3 scripts/run_walkforward_universe.py` (writes `walkforward_universe_latest.json` + `walkforward_universe_runs.jsonl`). Timer: [deploy/systemd/rlm-walkforward-universe.timer](deploy/systemd/rlm-walkforward-universe.timer). Legacy: `python3 scripts/run_walkforward.py` |
| Calibrate next-regime top-1 probabilities | `python3 scripts/fit_regime_transition_calibration.py --universe --regime-family hmm` → `data/processed/regime_transition_calibration.json` (optional `RLM_TRANSITION_CALIBRATION` path). Requires dev deps (`scikit-learn`). Built-in Markov/HMM transition fields = transition-law calibration (smooth + one-step with filtered probs); optional JSON = isotonic post-hoc on top-1 probs from forecast CSVs — not a full long-OOS regime-change calibration (use walk-forward JSONL for heavier scaling). |
| HMM forecast | `python3 scripts/run_forecast_pipeline.py --use-hmm --hmm-states 6 --no-vix` |
| Forecast (no Kronos) | `python3 scripts/run_forecast_pipeline.py --no-kronos --no-vix` |
| Backtest (no Kronos) | `python3 scripts/run_backtest.py --no-kronos --no-vix` |
| Fine-tune Kronos | `python3 scripts/finetune_kronos.py --symbol SPY --epochs 10` |
| Dashboard (Next.js) | `cd dashboard && npm install && npm run dev` (localhost:3000). Pages: `/` overview, **`/trading`** options/equities operating view (reads `universe_trade_plans.json`, `trade_log.csv`, `equity_trade_log.csv`, `data/challenge/state.json`). On VPS: `.\scripts\deploy_dashboard.ps1` — builds, starts via PM2. Access at `http://<host>:3000/`. |
| Master + Telegram (one process) | `python3 scripts/run_master.py --telegram-bot` — embeds the Telegram process; do not also run `scripts/rlm_telegram_bot.py`. Systemd: [deploy/rlm-master-telegram.service.example](deploy/rlm-master-telegram.service.example). Steps: [deploy/QUICK-ACTIVATE.txt](deploy/QUICK-ACTIVATE.txt). Telegram-only setup: [deploy/TELEGRAM_SETUP.txt](deploy/TELEGRAM_SETUP.txt). **VPS:** enable only **one** of `regime-locus-master`, `rlm-master-telegram`, or `rlm-master-trader` for the trading loop (unit files use `Conflicts=`); use standalone `rlm-telegram` for alerts without a second master. |
| Hermes AI crew (systemd `regime-locus-crew`) | `pip install -e ".[hermes]"` then `python3 scripts/run_crew.py`. Uses `RLM_HERMES_BASE_URL` (default `http://127.0.0.1:11434/v1`), `RLM_HERMES_MODEL`, optional `RLM_HERMES_SKIP_MEMORY=1`. RLM tools register as Hermes toolset `rlm`. Skill templates: `hermes_skills/`. Flow: **pipeline health** → **regime research** → **commander**. |
| Host watchdog (`rlm-host-watchdog.service`) | Separate from Hermes: `scripts/rlm_enterprise_watchdog.py`. Unit template: [deploy/linux/rlm-host-watchdog.service](deploy/linux/rlm-host-watchdog.service) (`/opt/Regime-Locus-Matrix` + `/opt/rlm-venv`). **NYSE window:** [deploy/linux/rlm-market-open.timer](deploy/linux/rlm-market-open.timer) Mon–Fri **09:00** ET and [rlm-market-close.timer](deploy/linux/rlm-market-close.timer) **16:30** ET run `rlm-market-hours-*.sh` to start/stop trading-heavy units (`rlm-master-trader`, `rlm-challenge-loop`, forecast timer, crew/bots per scripts); Hermes crew + systems Telegram usually stay enabled 24/7. Optional health JSON: `HOST_WATCHDOG_RLM_HEALTH=1` (legacy `SCOTTY_RLM_HEALTH`). Offline advisory: `scripts/rlm_offline_advisory.py` + `scripts/rlm_advisory_hook.py`. |
| Ship changes to Hostinger VPS | From repo root (Windows): **commit**, then `.\scripts\deploy_vps.ps1` — runs `git push origin main`, SSH `git pull` on `/opt/Regime-Locus-Matrix`, then restarts **active** systemd units from the script list (includes `regime-locus-crew`, host watchdog, systems-control telegram when those units exist and are running). Override with `-SystemdUnits` or env `VPS_SYSTEMD_UNITS`. If the server has local edits, use `-StashOnVpsBeforePull`. Defaults: `root@2.24.28.77`; override with env `VPS_HOST`, `VPS_USER`, `VPS_REPO`. |

### Agent note: deploy after edits

When you change this codebase in a session, **commit** the intended files, then **run `.\scripts\deploy_vps.ps1`** from the repo root. That script **`git push origin main`**, SSH **`git pull`** on the VPS repo path, then **`systemctl restart`** on **active** units from the deploy list (override with `-SystemdUnits` or `VPS_SYSTEMD_UNITS`). Require a **clean** working tree before the push step (stash or drop unrelated edits first). Skip only when the user explicitly opts out.

### Agent note: VPS and GPU — operate directly

**Do not** defer Hostinger VPS work to the user (no “you should SSH and…” handoffs). Unless they explicitly say they will handle infra themselves, **you** run commands, push fixes, and validate.

**VPS (default `root@2.24.28.77`, repo `/opt/Regime-Locus-Matrix`, venv often `/opt/rlm-venv`):**

- Use non-interactive SSH (`ssh -o BatchMode=yes`) from the dev machine when keys are configured; honour `VPS_HOST`, `VPS_USER`, `VPS_REPO` if set.
- After code or unit-template changes: **`deploy_vps.ps1`**, then if **`deploy/linux/*.service`** changed, **reinstall units on the server** (sed ` @INSTALL_ROOT@` → repo path into `/etc/systemd/system/`, `systemctl daemon-reload`, restart affected units).
- Diagnose with `systemctl status/list-units`, `journalctl`, log tails, `df`/`free`, and masked `grep` of `.env` (never print secrets).
- Edit remote `.env` via small scripts checked into `scripts/` (e.g. migrations), backups, and `systemctl restart` — not by telling the user to paste keys.
- Use Hostinger MCP (`VPS_getVirtualMachinesV1`, `VPS_getMetricsV1`, etc.) when cloud-side state matters.

**GPU:**

- Confirm **where** training or inference is supposed to run (this VPS is usually **CPU-only** KVM). Do not assume CUDA on the server.
- For local/GPU hosts: check `nvidia-smi`, `torch.cuda.is_available()`, Ollama GPU use, and env (`CUDA_VISIBLE_DEVICES`) as part of troubleshooting — run the checks yourself when you have shell access.
- Kronos defaults are **CPU-viable**; GPU is an optimization, not a requirement for “correct” behaviour.

**Exception:** If SSH or MCP auth fails or the user has no deploy path, say so once and state what is blocked — still avoid generic “run these steps” without attempting them from the environment you have.

### Gotchas

- Use `python3` (not `python`); the VM has Python 3.12 but no `python` symlink.
- `pip install` puts scripts in `~/.local/bin`; ensure `PATH` includes it (`export PATH="$HOME/.local/bin:$PATH"`).
- `--no-vix` skips yfinance VIX/VVIX download; use it for offline/isolated runs. `run_walkforward.py` does not accept `--no-vix`.
- `black --check` needs `--target-version py312` on Python 3.12 to avoid AST parse warnings for newer syntax.
- Pre-existing lint issues (133 ruff errors, 34 black reformats) exist in the repo; they are not regressions.
- External services (IBKR, Massive API, Massive S3) are optional; all unit tests use mocks/synthetic data. Set keys in `.env` (see `.env.example`) only when testing live data flows.
- `pyproject.toml` defines all config: pytest paths, ruff/black/mypy settings, and optional dependency groups (`ibkr`, `datalake`, `flatfiles`).
- Kronos (torch) is a **core** dependency (~2GB). The default model (`NeoQuasar/Kronos-mini`, 4.1M params) runs on CPU. First run downloads weights from HuggingFace. Pass `--no-kronos` to forecast/backtest scripts to skip Kronos entirely. Kronos config lives in `configs/default.yaml` under the `kronos:` block.
