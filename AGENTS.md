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

### Gotchas

- Use `python3` (not `python`); the VM has Python 3.12 but no `python` symlink.
- `pip install` puts scripts in `~/.local/bin`; ensure `PATH` includes it (`export PATH="$HOME/.local/bin:$PATH"`).
- `--no-vix` skips yfinance VIX/VVIX download; use it for offline/isolated runs. `run_walkforward.py` does not accept `--no-vix`.
- `black --check` needs `--target-version py312` on Python 3.12 to avoid AST parse warnings for newer syntax.
- Pre-existing lint issues (133 ruff errors, 34 black reformats) exist in the repo; they are not regressions.
- External services (IBKR, Massive API, Massive S3) are optional; all unit tests use mocks/synthetic data. Set keys in `.env` (see `.env.example`) only when testing live data flows.
- `pyproject.toml` defines all config: pytest paths, ruff/black/mypy settings, and optional dependency groups (`ibkr`, `datalake`, `flatfiles`).
