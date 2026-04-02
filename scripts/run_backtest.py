from __future__ import annotations

import pandas as pd

from rlm.backtest.engine import BacktestEngine
from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.pipeline import ForecastPipeline
from rlm.types.forecast import ForecastConfig


def main() -> None:
    bars = pd.read_csv("data/raw/sample_bars.csv", parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").set_index("timestamp")

    chain = pd.read_csv("data/raw/sample_option_chain.csv", parse_dates=["timestamp", "expiry"])

    features = FactorPipeline().run(bars)
    features = ForecastPipeline(
        config=ForecastConfig(
            drift_gamma_alpha=0.65,
            sigma_floor=1e-4,
            direction_neutral_threshold=0.3,
        ),
        move_window=100,
        vol_window=100,
    ).run(features)

    engine = BacktestEngine(
        initial_capital=100_000.0,
        contract_multiplier=100,
        strike_increment=5.0,
        underlying_symbol="SPY",
        quantity_per_trade=1,
    )

    equity_frame, trades_frame, summary = engine.run(features, chain)

    print("Backtest summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    equity_frame.to_csv("data/processed/backtest_equity.csv")
    trades_frame.to_csv("data/processed/backtest_trades.csv", index=False)


if __name__ == "__main__":
    main()
