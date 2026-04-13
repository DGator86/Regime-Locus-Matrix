# Regime Locus Matrix

A production-oriented, options-native quantitative engine. RLM ingests equity OHLCV from IBKR and options data from Massive into a Parquet/DuckDB lake, detects market regimes (HMM + Markov-switching), enriches signals with microstructure (GEX surfaces, IV surfaces, full Greeks), forecasts via the Kronos foundation model as a pure sensor, applies ROEE dynamic sizing and policy, and supports live/paper execution with continuous monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Ingestion                       │
│   IBKR stocks ──► Parquet/DuckDB lake ◄── Massive options  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                       Factor Pipeline                       │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │   Liquidity  │  │  Kronos Factors │  │ Microstructure│  │
│  │  & Orderflow │  │  (return / range│  │  GEX surface  │  │
│  │   Factors    │  │   / dispersion) │  │  IV surface   │  │
│  └──────────────┘  └────────┬────────┘  └───────────────┘  │
└───────────────────────────┬─┴───────────────────────────────┘
                            │  one scalar forecast per bar
┌───────────────────────────▼─────────────────────────────────┐
│                     Regime Detection                        │
│   HMM ──► MTF blend ──► binary regime gate                  │
│   Markov-switching                    ▲                     │
│   Kronos regime confidence ───────────┘                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      ROEE Policy                            │
│   Dynamic Kelly sizing · confidence gating · combo orders  │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   Backtest / Walk-forward          Live / Paper execution
   (BacktestEngine)                 (IBKR combo placement)
```

**Design principle:** Kronos is a pure sensor — one scalar return forecast plus a regime-agreement score. It never fuses with the HMM; the regime layer gates it with a binary pass/fail. ROEE is the single decision layer.

## FullRLMPipeline — one-line entry point

`FullRLMPipeline` (`src/rlm/pipeline.py`) is the single orchestrator that wires the entire flow above into one composable object:

```python
import pandas as pd
from rlm.pipeline import FullRLMPipeline, FullRLMConfig
from rlm.roee.pipeline import ROEEConfig

# Defaults: HMM(6) + Kronos overlay + ROEE
bars = pd.read_csv("data/raw/bars_SPY.csv", parse_dates=["timestamp"])
result = FullRLMPipeline().run(bars)
print(result.policy_df[["roee_action", "roee_strategy", "roee_size_fraction"]].tail())

# Custom config — dynamic Kelly sizing + backtest
cfg = FullRLMConfig(
    regime_model="hmm",
    hmm_states=6,
    use_kronos=True,
    roee_config=ROEEConfig(use_dynamic_sizing=True, max_kelly_fraction=0.2),
    run_backtest=True,
)
result = FullRLMPipeline(cfg).run(bars, option_chain_df=chain)
print(result.backtest_metrics)
```

`PipelineResult` bundles `factors_df`, `forecast_df`, `policy_df`, and optionally `backtest_trades`, `backtest_equity`, `backtest_metrics`.

---

## Kronos Foundation Model

Kronos is a time-series foundation model wired into RLM as a composable forecast factor. It is **on by default**; pass `--no-kronos` to disable the overlay.

### What Kronos produces

| Factor | Type | Description |
|--------|------|-------------|
| `kronos_return_forecast` | Direction | Predicted mean return over `pred_len` bars |
| `kronos_range_forecast` | Volatility | Predicted high–low range (volatility proxy) |
| `kronos_path_dispersion` | Volatility | Spread across sampled paths (uncertainty signal) |
| `kronos_confidence` | Overlay | Agreement between predicted and current regime |
| `kronos_regime_agreement` | Overlay | Scalar gate: 1 if regimes agree, 0 if conflict |
| `kronos_forecast_return` | Overlay | Final blended return signal |
| `kronos_forecast_vol` | Overlay | Final blended volatility signal |
| `kronos_transition_flag` | Overlay | 1 when a regime transition is predicted |

### Installation

```bash
pip install -e ".[kronos]"
# requires: torch>=2.0, einops>=0.8, huggingface-hub>=0.33, safetensors>=0.6
```

### Key classes

| Class | Location | Role |
|-------|----------|------|
| `KronosFactorCalculator` | `src/rlm/factors/kronos_factors.py` | Plugs into `FactorPipeline`; produces the three base factors |
| `RLMKronosPredictor` | `src/rlm/kronos/predictor.py` | Lazy-loading adapter; handles multi-sample path generation |
| `KronosRegimeConfidence` | `src/rlm/kronos/regime_confidence.py` | Derives `kronos_confidence` and `kronos_regime_agreement` |
| `KronosConfig` | `src/rlm/kronos/config.py` | Loaded from `configs/default.yaml`; controls `model_name`, `pred_len`, `sample_count`, `temperature`, `regime_confidence_weight` |
| `KronosForecastPipeline` | `src/rlm/forecasting/kronos_forecast.py` | Standalone Kronos forecast pipeline |

### Kronos CLI

```bash
# Standalone Kronos forecast pipeline
python scripts/run_kronos_pipeline.py --symbol SPY --sample-count 20

# With regime overlay (HMM-gated)
python scripts/run_kronos_pipeline.py --symbol SPY --use-hmm --hmm-states 6

# Fine-tune Kronos on local lake data
python scripts/finetune_kronos.py --symbol SPY

# Full forecast pipeline — Kronos on by default
python scripts/run_forecast_pipeline.py --symbol SPY --use-hmm

# Disable Kronos overlay (pure HMM/Markov path)
python scripts/run_forecast_pipeline.py --symbol SPY --use-hmm --no-kronos
```

### ROEE integration

`ROEEConfig` exposes two Kronos-specific knobs:

```python
ROEEConfig(
    kronos_confidence_weight=0.4,   # how much Kronos confidence modulates sizing
    kronos_transition_penalty=0.3,  # size haircut when kronos_transition_flag=1
)
```

---

## Microstructure Layer

The microstructure layer provides real-time Greeks, GEX surfaces, and IV surfaces via async IBKR collectors writing into a DuckDB-backed Parquet lake. All microstructure factors are optional and plug into `FactorPipeline.extra_calculators`.

### Installation

```bash
pip install -e ".[microstructure]"
# requires: duckdb>=0.10, pyarrow>=14.0
```

### Lake layout

```text
data/microstructure/
  underlying/{SYM}/1s/          ← 5s OHLCV bars (UnderlyingCollector)
  options/{SYM}/
    greeks_snapshots/            ← full 13-Greek surface, every 5 s
    derived/
      gex_surface/               ← net dealer GEX by strike & expiry
      iv_surface/                ← cubic-interpolated IV grid
```

### Key classes

| Class | Location | Role |
|-------|----------|------|
| `UnderlyingCollector` | `src/rlm/microstructure/collectors/underlying.py` | Async IBKR 5 s bar streamer |
| `OptionsCollector` | `src/rlm/microstructure/collectors/options.py` | Async Greek surface snapshots |
| `full_greeks_row()` | `src/rlm/microstructure/calculators/greeks.py` | Black-Scholes: all 13 Greeks + `solve_iv()` |
| `build_gex_surface_from_df()` | `src/rlm/microstructure/calculators/gex.py` | Net dealer GEX; `gex_flip_level()` |
| `build_iv_surface()` | `src/rlm/microstructure/calculators/iv_surface.py` | Cubic-interpolated IV grid; `query_iv_surface()` |
| `MicrostructureDB` | `src/rlm/microstructure/database/query.py` | DuckDB query layer; `microstructure_regime_context()` |
| `GEXFactors` | `src/rlm/microstructure/factors/gex_factors.py` | Factor calculator: `gex_net_total`, `gex_flip_distance`, … |
| `IVSurfaceFactors` | `src/rlm/microstructure/factors/iv_surface_factors.py` | Factor calculator: `iv_atm_30d`, `iv_skew_25d`, `iv_term_ratio`, … |

### GEX factors

| Column | Description |
|--------|-------------|
| `gex_net_total` | Aggregate signed dealer gamma exposure |
| `gex_sign` | Sign of net GEX (+1 / -1) |
| `gex_normalized` | Net GEX normalized by open interest |
| `gex_flip_distance` | Distance from spot to nearest GEX flip level |
| `gex_call_put_ratio` | Call GEX / Put GEX |

### IV surface factors

| Column | Description |
|--------|-------------|
| `iv_atm_30d` | ATM implied vol at 30-day tenor |
| `iv_skew_25d` | 25-delta put IV minus 25-delta call IV |
| `iv_term_ratio` | Front-month IV / back-month IV |
| `iv_surface_change` | Rolling change in ATM IV |
| `iv_vol_of_vol` | Realized volatility of IV changes |

### Microstructure CLI

```bash
# Start async collectors (underlying bars + Greek snapshots)
python scripts/run_microstructure_collectors.py --symbol SPY --interval 5 --max-dte 60

# Bars only (no options)
python scripts/run_microstructure_collectors.py --symbol SPY --no-options

# Batch build GEX & IV surfaces from stored snapshots
python scripts/build_microstructure_surfaces.py --symbol SPY

# Query the DuckDB lake directly
python -c "
from rlm.microstructure.database.query import MicrostructureDB
db = MicrostructureDB()
ctx = db.microstructure_regime_context('SPY')
print(ctx)
"
```

---

## Forecasting

### Regime and forecast layers

RLM remains a deterministic and explicit options-native engine. Optional regime layers include the original HMM overlay and a Markov-switching volatility model; the Kronos foundation model overlays a regime-confidence signal on top of both. Options enrichment derives local-surface-style features from the chain.

- `ForecastPipeline` is unchanged for deterministic operation.
- `ProbabilisticForecastPipeline` adds:
  - `forecast_return_lower`, `forecast_return_median`, `forecast_return_upper`
  - `forecast_uncertainty`, `realized_vol`, `forecast_source`
  - optional loading of an offline quantile model artifact trained by `scripts/train_probabilistic_model.py`
- `HybridForecastPipeline` augments output with:
  - `hmm_probs` (**forward-filtered** probabilities P(z_t | x_{1:t}) — no smoothing lookahead within the run dataframe)
  - `hmm_state` (argmax of filtered probabilities per bar)
  - `hmm_state_label` (mode `regime_key` label per state, from IS fit)
- `HybridMarkovForecastPipeline` adds:
  - `markov_probs` (filtered regime probabilities from a Markov-switching model fit on returns)
  - `markov_state`, `markov_state_label`
  - `markov_reference_col` metadata so downstream diagnostics know which return stream drove the fit
- Option-chain enrichment supports SVI-derived features:
  - `atm_forward_iv`, `surface_skew`, `surface_convexity`, `surface_fit_error`
- Dynamic sizing: `ROEEConfig(use_dynamic_sizing=True)` or `--dynamic-sizing`
- Transaction costs: extra friction via `LifecycleConfig.transaction_cost_config`
- Walk-forward: purging and regime-aware training-window expansion via `--purge-bars` / `--regime-aware`
- Backtests (`BacktestEngine`, `run_backtest.py`, `run_walkforward.py`): `--use-hmm` / `--use-markov` applies the full `ROEEConfig` path row-by-row
- `scripts/calibrate_regime_models.py` runs a weekly champion/challenger tournament; promotes the higher-Sharpe winner to `data/processed/live_regime_model.json`

### Example commands

```bash
# --- Kronos (default-on) ---
python scripts/run_forecast_pipeline.py --use-hmm --hmm-states 6
python scripts/run_forecast_pipeline.py --use-hmm --no-kronos          # disable Kronos overlay
python scripts/run_kronos_pipeline.py --symbol SPY --sample-count 20   # standalone Kronos

# --- HMM / Markov ---
python scripts/run_forecast_pipeline.py --use-markov --markov-states 3
python scripts/run_forecast_pipeline.py --probabilistic --model-path models/probabilistic_forecast.json

# --- ROEE ---
python scripts/run_roee_pipeline.py --use-hmm --hmm-states 6
python scripts/run_roee_pipeline.py --use-markov --markov-states 3 --dynamic-sizing
python scripts/run_roee_pipeline.py --probabilistic --dynamic-sizing

# --- Walk-forward / backtest ---
python scripts/run_walkforward.py --use-markov --probabilistic --dynamic-sizing --purge-bars 5 --regime-aware
python scripts/run_backtest.py --use-hmm --probabilistic --dynamic-sizing --hmm-states 6

# --- Training ---
python scripts/train_probabilistic_model.py --symbol SPY --out models/probabilistic_forecast.json
```

Batch ROEE labelling (does not run the backtest engine) remains in `apply_roee_policy` in `src/rlm/roee/pipeline.py`.

See `configs/default.yaml` for all tunable parameters.

---

## Data providers: IBKR stocks, Massive options

Recommended split:

- **Stocks (OHLCV bars):** Interactive Brokers via `rlm.data.ibkr_stocks.fetch_historical_stock_bars`. Requires [TWS or IB Gateway](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/) with API sockets enabled. Install: `pip install -e ".[ibkr]"`. Configure optional `.env` keys: `IBKR_HOST` (default `127.0.0.1`), `IBKR_PORT` (paper TWS `7497`, live `7496`; Gateway paper `4002`, live `4001`), `IBKR_CLIENT_ID` (default `1`). Smoke: `python scripts/fetch_ibkr.py SPY --duration "5 D" --bar-size "1 day"`.
- **Options (chains, snapshots, Greeks):** Massive only — `massive_option_chain_from_client`, `option_chain_snapshot`, etc.
- **Options (bulk history for research/backtests):** [Massive Flat Files](https://massive.com/docs/flat-files/quickstart) — gzip CSV over an S3-compatible endpoint (`MASSIVE_S3_*` credentials in `.env`). Use `scripts/ingest_massive_flatfiles_options.py` to land Parquet under `data/options/{SYM}/flatfiles/...`.

RLM does **not** use Massive for stocks; all equity OHLCV comes from **IBKR** (or demo/synthetic paths).

### Data lake (repeatable Parquet pulls)

Layout (gitignored by default):

```text
data/stocks/{SYM}/1d/*.parquet
data/stocks/{SYM}/1m/*.parquet
data/options/{SYM}/contracts/*.parquet
data/options/{SYM}/bars_1d/*.parquet
data/options/{SYM}/bars_1m/*.parquet
data/options/{SYM}/trades/*.parquet
data/options/{SYM}/quotes/*.parquet
data/options/{SYM}/flatfiles/{trades|quotes|day_aggs|minute_aggs}/{YYYY-MM-DD}.parquet
data/microstructure/underlying/{SYM}/1s/*.parquet
data/microstructure/options/{SYM}/greeks_snapshots/*.parquet
data/microstructure/options/{SYM}/derived/{gex_surface|iv_surface}/*.parquet
```

Install: `pip install -e ".[ibkr,datalake]"` (`pyarrow` for Parquet). For flat-file ingestion add `flatfiles` (`boto3` + `pyarrow`). For microstructure: `pip install -e ".[microstructure]"`.

| Step | Script |
|------|--------|
| 1 IBKR stocks | `python scripts/fetch_ibkr_stock_parquet.py SPY --duration "2 Y" --bar-size "1 day" --interval 1d` |
| 2 Massive contracts | `python scripts/fetch_massive_contracts.py --underlying SPY` |
| 3 Option bars | `python scripts/fetch_massive_option_bars.py --option-ticker O:... --underlying-path SPY --from YYYY-MM-DD --to YYYY-MM-DD` |
| 4 Quotes / trades | `fetch_massive_option_quotes.py` / `fetch_massive_option_trades.py` |
| 5 Options flat files (bulk) | `python scripts/ingest_massive_flatfiles_options.py --dataset trades --underlying SPY --from-date 2025-06-01 --to-date 2025-06-01` |
| 6 Microstructure collectors | `python scripts/run_microstructure_collectors.py --symbol SPY --interval 5` |
| 7 Build GEX & IV surfaces | `python scripts/build_microstructure_surfaces.py --symbol SPY` |
| Flat files S3 check | `python scripts/diagnose_massive_flatfiles_s3.py` |
| All-in-one | `python scripts/run_data_lake_pipeline.py --symbols SPY,QQQ` |

**Rule:** underlyings and stock history → **IBKR**; broad historical options tape/aggs → **Massive Flat Files**; contract metadata, targeted aggs/trades/quotes → **Massive REST**; real-time Greeks/GEX/IV → **microstructure collectors**. Strategy code reads **local Parquet** — not live APIs during backtests.

Optional **ib_insync** reference: `pip install -e ".[ib-insync]"` and `python scripts/examples/ib_insync_fetch_stock_example.py`.

**Live / paper option combos (IBKR):** `rlm.execution` resolves each leg with `reqContractDetails`, builds a **BAG**, and calls `placeOrder`. CLI: `python scripts/ibkr_place_roee_combo.py --spec combo.json` (default `transmit=False`). Live ports **7496** / **4001** require `--acknowledge-live`. Export a spec: `python scripts/run_decision_with_chain.py ... --write-ibkr-spec data/processed/ibkr_combo_spy.json`.

**Universe scan (market open):** `python scripts/analyze_universe_live.py` loops `LIQUID_UNIVERSE` (Mag 7 + SPY + QQQ) with IBKR daily history → factors → forecast → `select_trade` on the latest bar.

**Universe + real options + risk plan + monitor:** `python scripts/run_universe_options_pipeline.py` → writes `data/processed/universe_trade_plans.json`. Then `python scripts/monitor_active_trade_plans.py --plans data/processed/universe_trade_plans.json --interval 120` polls Massive mids and prints `ACTION: TAKE_PROFIT|HARD_STOP|TRAILING_STOP`.

**Single command (pipeline + one monitor pass):** `python scripts/run_everything.py`. Continuous: `python scripts/run_everything.py --follow --interval 120`.

**Full paper stack:** set `IBKR_PORT=7497` (or `4002`), then `python scripts/run_everything.py --full-paper --interval 120`.

### Local file layout (wired defaults)

All CLIs resolve paths from the **repository root** and use `--symbol` (default `SPY`) for `data/raw/bars_{SYMBOL}.csv` and `data/raw/option_chain_{SYMBOL}.csv` unless you pass `--bars` / `--chain` / `--out`.

| Step | Command |
|------|---------|
| Equity history (IBKR) | `python scripts/build_rolling_backtest_dataset.py --fetch-ibkr --symbol SPY --start 2022-01-01` → `bars_SPY.csv`; synthetic chain + manifest |
| Demo (no IBKR) | `python scripts/build_rolling_backtest_dataset.py --demo` |
| Append real option snapshot | `python scripts/append_option_snapshot.py --symbol SPY --as-of YYYY-MM-DD --replace-same-day` |
| Factors / forecast / ROEE | `python scripts/build_features.py`, `run_forecast_pipeline.py`, `run_roee_pipeline.py` — merge `option_chain_{SYMBOL}.csv` into bars, attach **^VIX / ^VVIX** via yfinance. Use `--no-vix` to skip macro. |
| Single backtest | `python scripts/run_backtest.py` (writes `data/processed/backtest_equity_{SYMBOL}.csv`) |
| 5m backtest | `python scripts/run_backtest_5m.py --demo --months 3 --symbol SPY --no-vix` |
| Walk-forward | `python scripts/run_walkforward.py` (writes `walkforward_*_{UNDERLYING}.csv`) |
| Tune forecast params | `python scripts/optimize_forecast_params.py --symbol SPY --trials 80 --objective composite` |
| Weekly regime tournament | `python scripts/calibrate_regime_models.py --symbol SPY --trials 24 --no-vix` or `scripts/weekly_regime_model_tournament.sh` |

---

## Multi-timeframe (MTF) workflow

All major pipelines support:

- `--mtf` — enable higher-timeframe factor augmentation via `MultiTimeframeEngine`
- `--higher-tfs` — comma-separated HTF rules (default: `1W,1M`)

Supported scripts: `calibrate_regime_models.py`, `run_forecast_pipeline.py`, `run_backtest.py`, `run_walkforward.py`, `run_data_lake_pipeline.py`.

### End-to-end MTF flow

```bash
# 1) Build/update local bars + chain
python3 scripts/build_rolling_backtest_dataset.py --fetch-ibkr --symbol SPY --start 2022-01-01

# 2) Pre-compute and validate HTF factor overlays
python3 scripts/run_forecast_pipeline.py --symbol SPY --mtf --higher-tfs 1W,1M --no-vix

# 3) Single backtest with HTF overlays
python3 scripts/run_backtest.py --symbol SPY --mtf --higher-tfs 1W,1M --no-vix

# 4) Walk-forward with HTF
python3 scripts/run_walkforward.py --symbol SPY --mtf --higher-tfs 1W,1M

# 5) Weekly regime calibration with HTF
python3 scripts/calibrate_regime_models.py --symbol SPY --trials 24 --mtf --higher-tfs 1W,1M --no-vix
```

Keep `--higher-tfs` consistent across scripts in the same experiment to avoid train/eval drift.

---

## Regime-stratified Kronos fine-tuning

`notebooks/regime_stratified_kronos_finetune.ipynb` walks through:

1. Loading bars from CSV or the MicrostructureDB DuckDB lake
2. Running `FactorPipeline` + `HybridForecastPipeline` to annotate HMM regime states
3. Splitting the bar series by `hmm_state` and fine-tuning one Kronos checkpoint per stratum
4. Comparing per-stratum validation losses
5. Exporting `regime_metadata.json` (consumed by the HF upload script)

Checkpoints are saved to `data/models/kronos/{SYMBOL}/regime_{state}/`.

```bash
# Run the notebook non-interactively
jupyter nbconvert --to notebook --execute \
    notebooks/regime_stratified_kronos_finetune.ipynb

# Upload to Hugging Face (dry-run first)
python scripts/upload_kronos_checkpoints_hf.py \
    --repo-id your-org/kronos-rlm-spy --symbol SPY --dry-run

python scripts/upload_kronos_checkpoints_hf.py \
    --repo-id your-org/kronos-rlm-spy --symbol SPY --private
```

---

## Installation summary

```bash
# Core
pip install -e "."

# With IBKR connectivity
pip install -e ".[ibkr]"

# With Parquet data lake
pip install -e ".[ibkr,datalake]"

# With Kronos foundation model (requires PyTorch)
pip install -e ".[kronos]"

# With microstructure layer (DuckDB, PyArrow)
pip install -e ".[microstructure]"

# Full production stack
pip install -e ".[kronos,microstructure,ibkr,datalake]"
```

Copy `.env.example` to `.env` and set `MASSIVE_API_KEY` (and optionally `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`). Do not commit `.env`.

---

## Massive API

Python client: `rlm.data.massive.MassiveClient` ([REST quickstart](https://massive.com/docs/rest/quickstart), [flat files quickstart](https://massive.com/docs/flat-files/quickstart), full index: [llms.txt](https://massive.com/docs/llms.txt)).

Smoke tests: `python scripts/fetch_massive.py SPY --endpoint option-snapshot`. Full connectivity: `python scripts/super_ping_data.py`.

**RLM option chain:** Map snapshot JSON with `massive_option_snapshot_to_normalized_chain` or fetch all pages via `massive_option_chain_from_client` / `collect_option_snapshot_pages`. Greeks and IV from the snapshot merge as `delta`, `gamma`, `theta`, `vega`, `iv`.

**Liquid watchlist:** `LIQUID_UNIVERSE` in `rlm.data.liquidity_universe` (Mag 7 + SPY + QQQ); `LIQUID_STOCK_UNIVERSE_10` adds `LIQUID_STOCK_EXTRAS` when you need ten single-names.
