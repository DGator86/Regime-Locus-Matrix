"""
Rolling degradation detection.

Markets change. A system that performed well in the IS window may silently
deteriorate OOS. This module computes rolling performance metrics over a
trailing trade window and flags when the system crosses degradation thresholds.

Typical usage:

    from rlm.backtest.degradation import compute_rolling_metrics, check_degradation

    rolling = compute_rolling_metrics(closed_trades_df, window=30)
    alert = check_degradation(rolling, expectancy_floor=-5.0, winrate_floor=0.3)
    if alert.degraded:
        reduce_size() or halt_trading()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DegradationThresholds:
    """Thresholds that trigger a degradation alert."""

    expectancy_floor: float = 0.0  # halt if rolling expectancy < this
    winrate_floor: float = 0.3  # alert if win rate < this
    drawdown_floor: float = -0.15  # alert if rolling drawdown < this (e.g. -15%)
    min_trades: int = 10  # require at least this many trades before alerting


@dataclass(frozen=True)
class DegradationAlert:
    degraded: bool
    reasons: list[str]
    rolling_expectancy: float
    rolling_winrate: float
    rolling_drawdown: float
    trade_count: int
    recommended_action: str  # "none" | "reduce_size" | "halt"


def compute_rolling_metrics(
    trades_frame: pd.DataFrame,
    window: int = 30,
    pnl_col: str = "pnl",
    pnl_pct_col: str = "pnl_pct",
) -> pd.DataFrame:
    """
    Compute trailing-window rolling performance metrics per trade.

    Returns a DataFrame with the same index as ``trades_frame`` plus columns:
        rolling_expectancy, rolling_winrate, rolling_drawdown,
        rolling_avg_return, rolling_avg_return_pct

    The window is backwards-looking only (no lookahead).
    """
    if trades_frame.empty or pnl_col not in trades_frame.columns:
        return pd.DataFrame()

    pnl = trades_frame[pnl_col]

    rolling_mean = pnl.rolling(window, min_periods=1).mean()
    rolling_winrate = (pnl > 0).astype(float).rolling(window, min_periods=1).mean()

    # Expectancy = E[PnL | window]
    # Approximate as rolling mean (signed average, accounts for magnitude).
    rolling_expectancy = rolling_mean.copy()

    # Rolling drawdown on cumulative PnL within the window.
    cum_pnl = pnl.cumsum()
    rolling_peak = cum_pnl.rolling(window, min_periods=1).max()
    rolling_drawdown = (cum_pnl - rolling_peak) / rolling_peak.abs().replace(0, np.nan)

    out = trades_frame.copy()
    out["rolling_expectancy"] = rolling_expectancy
    out["rolling_winrate"] = rolling_winrate
    out["rolling_drawdown"] = rolling_drawdown
    out["rolling_avg_return"] = rolling_mean

    if pnl_pct_col in trades_frame.columns:
        out["rolling_avg_return_pct"] = trades_frame[pnl_pct_col].rolling(window, min_periods=1).mean()

    return out


def check_degradation(
    rolling_metrics: pd.DataFrame,
    thresholds: DegradationThresholds | None = None,
) -> DegradationAlert:
    """
    Evaluate the latest rolling window snapshot against degradation thresholds.

    Call after each completed trade (or at end-of-day) and act on the result:
        - "none"        → system healthy, full size
        - "reduce_size" → one threshold breached, cut size by 50%
        - "halt"        → multiple thresholds breached or expectancy deeply negative
    """
    t = thresholds or DegradationThresholds()

    if rolling_metrics.empty:
        return DegradationAlert(
            degraded=False,
            reasons=[],
            rolling_expectancy=np.nan,
            rolling_winrate=np.nan,
            rolling_drawdown=np.nan,
            trade_count=0,
            recommended_action="none",
        )

    last = rolling_metrics.iloc[-1]
    trade_count = int(len(rolling_metrics))
    expectancy = float(last.get("rolling_expectancy", np.nan))
    winrate = float(last.get("rolling_winrate", np.nan))
    drawdown = float(last.get("rolling_drawdown", np.nan))

    reasons: list[str] = []

    if trade_count < t.min_trades:
        return DegradationAlert(
            degraded=False,
            reasons=[f"Insufficient trades ({trade_count} < {t.min_trades})"],
            rolling_expectancy=expectancy,
            rolling_winrate=winrate,
            rolling_drawdown=drawdown,
            trade_count=trade_count,
            recommended_action="none",
        )

    if np.isfinite(expectancy) and expectancy < t.expectancy_floor:
        reasons.append(f"Rolling expectancy {expectancy:.2f} < floor {t.expectancy_floor:.2f}")
    if np.isfinite(winrate) and winrate < t.winrate_floor:
        reasons.append(f"Rolling win rate {winrate:.2%} < floor {t.winrate_floor:.2%}")
    if np.isfinite(drawdown) and drawdown < t.drawdown_floor:
        reasons.append(f"Rolling drawdown {drawdown:.2%} < floor {t.drawdown_floor:.2%}")

    degraded = len(reasons) > 0
    if not degraded:
        action = "none"
    elif len(reasons) >= 2:
        action = "halt"
    else:
        action = "reduce_size"

    return DegradationAlert(
        degraded=degraded,
        reasons=reasons,
        rolling_expectancy=expectancy,
        rolling_winrate=winrate,
        rolling_drawdown=drawdown,
        trade_count=trade_count,
        recommended_action=action,
    )
