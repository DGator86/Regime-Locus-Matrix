"""Dataset builders for backtests and walk-forward runs."""

from rlm.datasets.backtest_data import (
    fetch_ibkr_daily_bars_range,
    rolling_window_manifest,
    synthetic_bars_demo,
    synthetic_option_chain_from_bars,
)
from rlm.datasets.paths import (
    DEFAULT_SYMBOL,
    backtest_equity_filename,
    backtest_trades_filename,
    rel_bars_csv,
    rel_forecast_features_csv,
    rel_features_csv,
    rel_option_chain_csv,
    rel_roee_policy_csv,
    walkforward_equity_filename,
    walkforward_summary_filename,
    walkforward_trades_filename,
)

__all__ = [
    "DEFAULT_SYMBOL",
    "backtest_equity_filename",
    "backtest_trades_filename",
    "fetch_ibkr_daily_bars_range",
    "rel_bars_csv",
    "rel_forecast_features_csv",
    "rel_features_csv",
    "rel_option_chain_csv",
    "rel_roee_policy_csv",
    "rolling_window_manifest",
    "synthetic_bars_demo",
    "synthetic_option_chain_from_bars",
    "walkforward_equity_filename",
    "walkforward_summary_filename",
    "walkforward_trades_filename",
]
