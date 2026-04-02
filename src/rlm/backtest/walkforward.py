from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlm.backtest.engine import BacktestEngine
from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.pipeline import ForecastPipeline
from rlm.types.forecast import ForecastConfig


@dataclass(frozen=True)
class WalkForwardConfig:
    is_window: int = 100
    oos_window: int = 50
    step_size: int = 50
    initial_capital: float = 100_000.0
    strike_increment: float = 1.0
    underlying_symbol: str = "SPY"
    quantity_per_trade: int = 1


def run_walkforward(
    *,
    bars: pd.DataFrame,
    option_chain: pd.DataFrame,
    forecast_config: ForecastConfig | None = None,
    wf_config: WalkForwardConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = wf_config or WalkForwardConfig()
    fc = forecast_config or ForecastConfig()

    bars = bars.sort_index().copy()
    all_equity = []
    all_trades = []
    window_summaries = []

    start = 0
    window_id = 0

    while start + cfg.is_window + cfg.oos_window <= len(bars):
        is_start = start
        is_end = start + cfg.is_window
        oos_end = is_end + cfg.oos_window

        is_bars = bars.iloc[is_start:is_end].copy()
        oos_bars = bars.iloc[is_end:oos_end].copy()

        # For now we only "fit" by computing rolling features from concatenated IS+OOS,
        # while evaluation uses only OOS rows.
        # Later this is where you'd freeze tuned params from IS.
        joined = pd.concat([is_bars, oos_bars], axis=0)

        feature_df = FactorPipeline().run(joined)
        feature_df = ForecastPipeline(
            config=fc,
            move_window=cfg.is_window,
            vol_window=cfg.is_window,
        ).run(feature_df)

        oos_features = feature_df.loc[oos_bars.index].copy()
        oos_chain = option_chain[option_chain["timestamp"].isin(oos_features.index)].copy()

        engine = BacktestEngine(
            initial_capital=cfg.initial_capital,
            strike_increment=cfg.strike_increment,
            underlying_symbol=cfg.underlying_symbol,
            quantity_per_trade=cfg.quantity_per_trade,
        )

        equity_frame, trades_frame, summary = engine.run(oos_features, oos_chain)

        if not equity_frame.empty:
            eq = equity_frame.copy()
            eq["window_id"] = window_id
            all_equity.append(eq)

        if not trades_frame.empty:
            tr = trades_frame.copy()
            tr["window_id"] = window_id
            all_trades.append(tr)

        summary_row = {
            "window_id": window_id,
            "is_start": str(is_bars.index[0]),
            "is_end": str(is_bars.index[-1]),
            "oos_start": str(oos_bars.index[0]),
            "oos_end": str(oos_bars.index[-1]),
        }
        summary_row.update(summary)
        window_summaries.append(summary_row)

        window_id += 1
        start += cfg.step_size

    equity_df = pd.concat(all_equity).sort_index() if all_equity else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_df = pd.DataFrame(window_summaries)

    return equity_df, trades_df, summary_df
