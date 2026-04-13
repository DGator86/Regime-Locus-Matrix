"""Repo-relative paths for IBKR bars, Massive option chains, and processed artifacts."""

from __future__ import annotations

DEFAULT_SYMBOL = "SPY"


def rel_bars_csv(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"data/raw/bars_{symbol.upper()}.csv"


def rel_option_chain_csv(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"data/raw/option_chain_{symbol.upper()}.csv"


def rel_features_csv(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"data/processed/features_{symbol.upper()}.csv"


def rel_forecast_features_csv(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"data/processed/forecast_features_{symbol.upper()}.csv"


def rel_roee_policy_csv(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"data/processed/roee_policy_{symbol.upper()}.csv"


def walkforward_equity_filename(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"walkforward_equity_{symbol.upper()}.csv"


def walkforward_trades_filename(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"walkforward_trades_{symbol.upper()}.csv"


def walkforward_summary_filename(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"walkforward_summary_{symbol.upper()}.csv"


def backtest_equity_filename(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"backtest_equity_{symbol.upper()}.csv"


def backtest_trades_filename(symbol: str = DEFAULT_SYMBOL) -> str:
    return f"backtest_trades_{symbol.upper()}.csv"


# ---------------------------------------------------------------------------
# Microstructure lake paths
# ---------------------------------------------------------------------------

_MICRO_ROOT = "data/microstructure"


def rel_micro_underlying_dir(symbol: str) -> str:
    """data/microstructure/underlying/{SYMBOL}/1s/"""
    return f"{_MICRO_ROOT}/underlying/{symbol.upper()}/1s"


def rel_micro_greeks_snapshots_dir(symbol: str) -> str:
    """data/microstructure/options/{SYMBOL}/greeks_snapshots/"""
    return f"{_MICRO_ROOT}/options/{symbol.upper()}/greeks_snapshots"


def rel_micro_gex_surface_dir(symbol: str) -> str:
    """data/microstructure/options/{SYMBOL}/derived/gex_surface/"""
    return f"{_MICRO_ROOT}/options/{symbol.upper()}/derived/gex_surface"


def rel_micro_iv_surface_dir(symbol: str) -> str:
    """data/microstructure/options/{SYMBOL}/derived/iv_surface/"""
    return f"{_MICRO_ROOT}/options/{symbol.upper()}/derived/iv_surface"


def rel_micro_gex_parquet(symbol: str, date: str) -> str:
    """data/microstructure/options/{SYMBOL}/derived/gex_surface/{SYMBOL}_{date}.parquet"""
    return f"{rel_micro_gex_surface_dir(symbol)}/{symbol.upper()}_{date}.parquet"


def rel_micro_iv_parquet(symbol: str, date: str) -> str:
    """data/microstructure/options/{SYMBOL}/derived/iv_surface/{SYMBOL}_{date}.parquet"""
    return f"{rel_micro_iv_surface_dir(symbol)}/{symbol.upper()}_{date}.parquet"
