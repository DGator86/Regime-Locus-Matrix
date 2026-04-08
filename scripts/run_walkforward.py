from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.backtest.walkforward import WalkForwardConfig, run_walkforward
from rlm.datasets.paths import (
    DEFAULT_SYMBOL,
    rel_bars_csv,
    rel_option_chain_csv,
    walkforward_equity_filename,
    walkforward_summary_filename,
    walkforward_trades_filename,
)
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.types.forecast import ForecastConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward backtest. Build inputs first: scripts/build_rolling_backtest_dataset.py"
    )
    p.add_argument("--use-hmm", action="store_true")
    p.add_argument("--hmm-states", type=int, default=6)
    p.add_argument("--use-markov", action="store_true", help="Use Markov-switching regime model.")
    p.add_argument(
        "--probabilistic", action="store_true", help="Use probabilistic forecast output."
    )
    p.add_argument(
        "--model-path", type=str, default=None, help="Optional quantile model artifact JSON."
    )
    p.add_argument("--dynamic-sizing", action="store_true", help="Enable Kelly/vol-target sizing.")
    p.add_argument(
        "--vault-uncertainty-threshold",
        type=float,
        default=0.03,
        help="Cut size when forecast 5th-95th range exceeds this return-width threshold.",
    )
    p.add_argument(
        "--vault-size-multiplier",
        type=float,
        default=0.5,
        help="Position-size multiplier to apply when the Vault rule triggers.",
    )
    p.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Ticker for default --bars/--chain paths (default {DEFAULT_SYMBOL})",
    )
    p.add_argument(
        "--bars",
        type=str,
        default=None,
        help="Bars CSV (default: data/raw/bars_{SYMBOL}.csv).",
    )
    p.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Option chain CSV (default: data/raw/option_chain_{SYMBOL}.csv).",
    )
    p.add_argument("--is-window", type=int, default=100)
    p.add_argument("--oos-window", type=int, default=50)
    p.add_argument("--step-size", type=int, default=50)
    p.add_argument(
        "--purge-bars", type=int, default=0, help="Purge bars between IS and OOS windows."
    )
    p.add_argument(
        "--regime-aware", action="store_true", help="Expand IS windows to cover OOS regimes."
    )
    p.add_argument(
        "--min-regime-train-samples",
        type=int,
        default=20,
        help="Minimum training samples for each OOS regime when --regime-aware.",
    )
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--strike-increment", type=float, default=5.0)
    p.add_argument(
        "--underlying",
        type=str,
        default=None,
        help="Engine symbol (default: same as --symbol)",
    )
    p.add_argument("--quantity", type=int, default=1, dest="quantity_per_trade")
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/processed",
        help="Directory for walkforward_*.csv outputs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sym = str(args.symbol).upper().strip()
    und = str(args.underlying).upper().strip() if args.underlying else sym
    bars_rel = args.bars or rel_bars_csv(sym)
    chain_rel = args.chain or rel_option_chain_csv(sym)
    bars_path = ROOT / bars_rel
    chain_path = ROOT / chain_rel
    if not bars_path.is_file():
        raise SystemExit(
            f"Bars file not found: {bars_path}\n"
            "Run: python scripts/build_rolling_backtest_dataset.py --demo\n"
            f"  or: python scripts/build_rolling_backtest_dataset.py --fetch-ibkr --symbol {sym} --start 2022-01-01"
        )
    if not chain_path.is_file():
        raise SystemExit(
            f"Chain file not found: {chain_path}\n"
            "Synthetic chain: python scripts/build_rolling_backtest_dataset.py (writes option_chain_* from bars)\n"
            "Real snapshots: python scripts/append_option_snapshot.py --symbol "
            f"{sym} --as-of YYYY-MM-DD --replace-same-day"
        )

    bars = pd.read_csv(bars_path, parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").set_index("timestamp")

    chain = pd.read_csv(chain_path, parse_dates=["timestamp", "expiry"])

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_df, trades_df, summary_df = run_walkforward(
        bars=bars,
        option_chain=chain,
        forecast_config=ForecastConfig(
            drift_gamma_alpha=0.65,
            sigma_floor=1e-4,
            direction_neutral_threshold=0.3,
        ),
        wf_config=WalkForwardConfig(
            is_window=args.is_window,
            oos_window=args.oos_window,
            step_size=args.step_size,
            initial_capital=args.initial_capital,
            strike_increment=args.strike_increment,
            underlying_symbol=und,
            quantity_per_trade=args.quantity_per_trade,
            use_dynamic_sizing=args.dynamic_sizing,
            vault_uncertainty_threshold=args.vault_uncertainty_threshold,
            vault_size_multiplier=args.vault_size_multiplier,
            purge_bars=args.purge_bars,
            regime_aware=args.regime_aware,
            min_regime_train_samples=args.min_regime_train_samples,
        ),
        use_hmm=args.use_hmm,
        hmm_config=HMMConfig(n_states=args.hmm_states) if args.use_hmm else None,
        use_markov=args.use_markov,
        markov_config=MarkovSwitchingConfig() if args.use_markov else None,
        use_probabilistic=args.probabilistic,
        probabilistic_model_path=args.model_path,
    )

    wf_sym = und
    equity_df.to_csv(out_dir / walkforward_equity_filename(wf_sym))
    trades_df.to_csv(out_dir / walkforward_trades_filename(wf_sym), index=False)
    summary_df.to_csv(out_dir / walkforward_summary_filename(wf_sym), index=False)

    print(summary_df)


if __name__ == "__main__":
    main()
