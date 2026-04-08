# Regime Locus Matrix

## Forecasting

### HMM Hybrid Layer

RLM remains a deterministic and explicit options-native engine. The optional HMM layer adds soft probabilities over the standardized factor space (`S_D`, `S_V`, `S_L`, `S_G`) and exposes persistence/transition-aware confidence for downstream sizing and gating.

- `ForecastPipeline` is unchanged for deterministic operation.
- `ProbabilisticForecastPipeline` adds:
  - `forecast_return_lower`, `forecast_return_median`, `forecast_return_upper`
  - `forecast_uncertainty`, `realized_vol`, `forecast_source`
  - optional loading of an offline quantile model artifact trained by `scripts/train_probabilistic_model.py`
- `HybridForecastPipeline` augments output with:
  - `hmm_probs` (**forward-filtered** probabilities P(z_t | x_{1:t}) — no smoothing lookahead within the run dataframe)
  - `hmm_state` (argmax of filtered probabilities per bar)
  - `hmm_state_label` (mode `regime_key` label per state, from IS fit)
- Dynamic sizing is now available through `ROEEConfig(use_dynamic_sizing=True)` or the CLI flag `--dynamic-sizing`; the backtest converts `size_fraction` into actual contract quantity rather than storing it as metadata only.
- Transaction costs now support an extra friction model on top of fill slippage and commissions via `LifecycleConfig.transaction_cost_config`.
- **Backtests** (`BacktestEngine`, `run_backtest.py`, `run_walkforward.py`): when you pass `--use-hmm`, the engine uses `ROEEConfig` so the same HMM confidence gate and size multiplier as `apply_roee_policy` apply row-by-row. Without `--use-hmm`, decisions use `select_trade` only (no HMM columns required).

Example usage:

```bash
python scripts/run_forecast_pipeline.py --use-hmm --hmm-states 6
python scripts/run_forecast_pipeline.py --probabilistic --model-path models/probabilistic_forecast.json
python scripts/run_roee_pipeline.py --use-hmm --hmm-states 6
python scripts/run_roee_pipeline.py --probabilistic --dynamic-sizing
python scripts/run_walkforward.py --use-hmm --probabilistic --dynamic-sizing --hmm-states 6
python scripts/run_backtest.py --use-hmm --probabilistic --dynamic-sizing --hmm-states 6
python scripts/train_probabilistic_model.py --symbol SPY --out models/probabilistic_forecast.json
```

Batch ROEE labelling (does not run the backtest engine) remains in `apply_roee_policy` in `src/rlm/roee/pipeline.py`.

See `configs/default.yaml` for a reference of tunable parameters.

## Data providers: IBKR stocks, Massive options

Recommended split:

- **Stocks (OHLCV bars):** Interactive Brokers via `rlm.data.ibkr_stocks.fetch_historical_stock_bars` — same columns as Massive bars (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `vwap`). Requires [TWS or IB Gateway](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/) with API sockets enabled. Install: `pip install -e ".[ibkr]"`. Configure optional `.env` keys: `IBKR_HOST` (default `127.0.0.1`), `IBKR_PORT` (paper TWS `7497`, live `7496`; Gateway paper `4002`, live `4001`), `IBKR_CLIENT_ID` (default `1`). Smoke: `python scripts/fetch_ibkr.py SPY --duration "5 D" --bar-size "1 day"`.
- **Options (chains, snapshots, Greeks):** Massive only — `massive_option_chain_from_client`, `option_chain_snapshot`, etc. Do not duplicate option chain plumbing on IB unless you explicitly need it later.
- **Options (bulk history for research/backtests):** [Massive Flat Files](https://massive.com/docs/flat-files/quickstart) — gzip CSV over an S3-compatible endpoint (`MASSIVE_S3_*` credentials in `.env`, not the REST key alone). Use `scripts/ingest_massive_flatfiles_options.py` to land Parquet under `data/options/{SYM}/flatfiles/...`. Keep REST + `fetch_massive_*.py` for incremental pulls, contract lookup, and narrow windows.

RLM does **not** use Massive for stocks; all equity OHLCV comes from **IBKR** (or demo/synthetic paths). Low-level `MassiveClient` still exposes stock endpoints for ad-hoc scripts, but pipelines and smoke checks treat Massive as **options-only**.

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
```

Install: `pip install -e ".[ibkr,datalake]"` ( **`pyarrow`** for Parquet). For flat-file ingestion add **`flatfiles`** (`boto3` + `pyarrow`). `MassiveClient` adds `option_contracts_reference`, `option_aggs_range`, `option_trades`, `option_quotes`.

| Step | Script |
|------|--------|
| 1 IBKR stocks | `python scripts/fetch_ibkr_stock_parquet.py SPY --duration "2 Y" --bar-size "1 day" --interval 1d` and `--interval 1m` for intraday |
| 2 Massive contracts | `python scripts/fetch_massive_contracts.py --underlying SPY` |
| 3 Option bars | `python scripts/fetch_massive_option_bars.py --option-ticker O:... --underlying-path SPY --from YYYY-MM-DD --to YYYY-MM-DD --timespan day` |
| 4 Quotes / trades | `fetch_massive_option_quotes.py` / `fetch_massive_option_trades.py` (narrow time windows) |
| 5 Options flat files (bulk) | `python scripts/ingest_massive_flatfiles_options.py --dataset trades --underlying SPY --from-date 2025-06-01 --to-date 2025-06-01` (add `--dry-run` to print S3 keys; confirm prefixes in dashboard if objects 404) |
| Flat files S3 check | `python scripts/diagnose_massive_flatfiles_s3.py` — if **ListObjects** works but **GetObject** returns **403**, your signing/URL layout is usually fine; ask [Massive](https://massive.com/docs/flat-files/quickstart) support to enable **object read** for your flat-file key (same class of issue seen with Polygon flat files and third-party S3 SDKs). |
| All-in-one | `python scripts/run_data_lake_pipeline.py --symbols SPY,QQQ` (optional `--option-tickers O:... --fetch-quotes --fetch-trades`) |

**Rule:** underlyings and stock history → **IBKR**; broad historical options tape/aggs → **Massive Flat Files**; contract metadata, targeted option aggs/trades/quotes → **Massive REST**. Strategy code should read **local Parquet** (or your CSV pipeline) — not live APIs during backtests.

Optional **ib_insync** reference: `pip install -e ".[ib-insync]"` and `python scripts/examples/ib_insync_fetch_stock_example.py` (standalone; core RLM uses **ibapi**).

**Live / paper option combos (IBKR):** `rlm.execution` resolves each leg with `reqContractDetails`, builds a **BAG**, and calls `placeOrder`. CLI: `python scripts/ibkr_place_roee_combo.py --spec combo.json` (default `transmit=False` — confirm or transmit in TWS). Live ports **7496** / **4001** require `--acknowledge-live`. Export a spec from a matched chain: `python scripts/run_decision_with_chain.py ... --write-ibkr-spec data/processed/ibkr_combo_spy.json`. **Universe scan (market open):** `python scripts/analyze_universe_live.py` loops `LIQUID_UNIVERSE` (Mag 7 + SPY + QQQ) with IBKR daily history → factors → forecast → `select_trade` on the latest bar (abstract σ-legs only).

**Universe + real options + risk plan + monitor:** `python scripts/run_universe_options_pipeline.py` → writes `data/processed/universe_trade_plans.json` (matched Massive quotes, entry debit, take-profit / hard-stop / trailing levels, `ibkr_combo_spec`). Then `python scripts/monitor_active_trade_plans.py --plans data/processed/universe_trade_plans.json --interval 120` polls Massive mids and prints `ACTION: TAKE_PROFIT|HARD_STOP|TRAILING_STOP` (does not auto-close at IBKR by default).

**Single command (pipeline + one monitor pass):** from repo root, `python scripts/run_everything.py`. Continuous monitoring: `python scripts/run_everything.py --follow --interval 120`.

**Full paper stack** (scan → plan → IBKR limit opens → Massive monitor → IBKR market closes on exit signals): set `IBKR_PORT=7497` (or `4002`), then `python scripts/run_everything.py --full-paper --interval 120`. Equivalent: `--paper-trade --paper-close --follow`. Dry runs: `--paper-dry-run` / `--paper-close-dry-run`.

### Local file layout (wired defaults)

All CLIs resolve paths from the **repository root** and use **`--symbol`** (default `SPY`) for `data/raw/bars_{SYMBOL}.csv` and `data/raw/option_chain_{SYMBOL}.csv` unless you pass `--bars` / `--chain` / `--out`.

| Step | Command |
|------|---------|
| Equity history (IBKR) | `python scripts/build_rolling_backtest_dataset.py --fetch-ibkr --symbol SPY --start 2022-01-01` → `bars_SPY.csv`; synthetic chain + manifest |
| Demo (no IBKR) | `python scripts/build_rolling_backtest_dataset.py --demo` |
| Append real option snapshot (Massive) | `python scripts/append_option_snapshot.py --symbol SPY --as-of YYYY-MM-DD --replace-same-day` → merges into `option_chain_SPY.csv` |
| Factors / forecast / ROEE | `python scripts/build_features.py`, `run_forecast_pipeline.py`, `run_roee_pipeline.py` — they **merge** `option_chain_{SYMBOL}.csv` into bars (dealer GEX/skew/term structure, ATM bid–ask) and attach **^VIX / ^VVIX** via yfinance when online. Use `--no-vix` to skip macro. |
| Single backtest | `python scripts/run_backtest.py` (writes `data/processed/backtest_equity_{SYMBOL}.csv`) |
| 5m backtest (e.g. 3 months) | `python scripts/run_backtest_5m.py --demo --months 3 --symbol SPY --no-vix` (synthetic 5m RTH bars + chain); with TWS: `--fetch-ibkr --months 3`. Writes `data/processed/backtest_equity_{SYM}_5m.csv` |
| Walk-forward | `python scripts/run_walkforward.py` (writes `walkforward_*_{UNDERLYING}.csv`; `--underlying` defaults to `--symbol`) |
| Tune forecast params | `python scripts/optimize_forecast_params.py --symbol SPY --trials 80 --objective composite` → prints top runs and writes `data/processed/forecast_param_search.json` |

## Massive API

Python client: `rlm.data.massive.MassiveClient` ([REST quickstart](https://massive.com/docs/rest/quickstart), [flat files quickstart](https://massive.com/docs/flat-files/quickstart), full index: [llms.txt](https://massive.com/docs/llms.txt)).

1. Copy `.env.example` to `.env` and set `MASSIVE_API_KEY` (do not commit `.env`).
2. Smoke tests (options):
   - `python scripts/fetch_massive.py SPY --endpoint option-snapshot`
   - Full connectivity: `python scripts/super_ping_data.py` (Massive **options** REST + OPRA S3 probe, IBKR equity, optional yfinance)

Equity bars: `python scripts/fetch_ibkr.py SPY` (see **Data providers**). For Massive, use `client.get("/v2/...")` or `client.get("/v3/...")` on **options** paths; helpers include `option_chain_snapshot`, `option_contracts_reference`, `option_aggs_range`, `get_by_url` for paginated `next_url`. Stock helpers (`stock_aggs_range`, `stock_trades`, …) exist on the client but are **not** part of RLM’s intended data split.

**RLM option chain:** Map snapshot JSON with `massive_option_snapshot_to_normalized_chain` or fetch all pages via `massive_option_chain_from_client` / `collect_option_snapshot_pages` in `rlm.data.massive_option_chain`. Greeks and implied volatility from the snapshot are merged into the normalized frame as `delta`, `gamma`, `theta`, `vega`, and `iv` when present.

**Equity bars in RLM:** Use IBKR (`rlm.data.ibkr_stocks`) — see **Data providers** above.

**Optional (not RLM default):** `rlm.data.massive_stocks` maps Massive stock aggs to bars; order-flow helpers (`stock_trades`, `aggregate_trade_flow_to_bars`, …) target Massive equity ticks. RLM’s split is **IBKR for stocks, Massive for options**; use those modules only if you deliberately pull equity from Massive.

**Liquid watchlist:** `LIQUID_UNIVERSE` in `rlm.data.liquidity_universe` (Magnificent 7 + SPY + QQQ) for batch pulls; `LIQUID_STOCK_UNIVERSE_10` adds `LIQUID_STOCK_EXTRAS` when you need ten single-names.
