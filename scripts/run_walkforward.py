from __future__ import annotations

import argparse

import pandas as pd

from rlm.backtest.walkforward import run_walkforward, WalkForwardConfig
from rlm.forecasting.hmm import HMMConfig
from rlm.types.forecast import ForecastConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-hmm", action="store_true")
    parser.add_argument("--hmm-states", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bars = pd.read_csv("data/raw/sample_bars.csv", parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").set_index("timestamp")

    chain = pd.read_csv("data/raw/sample_option_chain.csv", parse_dates=["timestamp", "expiry"])

    equity_df, trades_df, summary_df = run_walkforward(
        bars=bars,
        option_chain=chain,
        forecast_config=ForecastConfig(
            drift_gamma_alpha=0.65,
            sigma_floor=1e-4,
            direction_neutral_threshold=0.3,
        ),
        wf_config=WalkForwardConfig(
            is_window=100,
            oos_window=50,
            step_size=50,
            initial_capital=100_000.0,
            strike_increment=5.0,
            underlying_symbol="SPY",
            quantity_per_trade=1,
        ),
        use_hmm=args.use_hmm,
        hmm_config=HMMConfig(n_states=args.hmm_states) if args.use_hmm else None,
    )

    equity_df.to_csv("data/processed/walkforward_equity.csv")
    trades_df.to_csv("data/processed/walkforward_trades.csv", index=False)
    summary_df.to_csv("data/processed/walkforward_summary.csv", index=False)

    print(summary_df)


if __name__ == "__main__":
    main()
