from __future__ import annotations

import numpy as np
import pandas as pd


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    std = r.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return float((r.mean() / std) * np.sqrt(periods_per_year))


def compute_win_rate(trade_pnls: pd.Series) -> float:
    t = trade_pnls.dropna()
    if len(t) == 0:
        return np.nan
    return float((t > 0).mean())


def compute_profit_factor(trade_pnls: pd.Series) -> float:
    t = trade_pnls.dropna()
    if len(t) == 0:
        return np.nan
    gross_profit = t[t > 0].sum()
    gross_loss = -t[t < 0].sum()
    if gross_loss <= 0:
        return np.nan
    return float(gross_profit / gross_loss)


def compute_expectancy(trade_pnls: pd.Series) -> float:
    """Win-rate × avg-win − loss-rate × avg-loss, in PnL units."""
    t = trade_pnls.dropna()
    if len(t) == 0:
        return np.nan
    wins = t[t > 0]
    losses = t[t < 0]
    win_rate = len(wins) / len(t)
    loss_rate = 1.0 - win_rate
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    return float(win_rate * avg_win + loss_rate * avg_loss)


def summarize_backtest(
    equity_frame: pd.DataFrame,
    trades_frame: pd.DataFrame,
) -> dict[str, float]:
    if equity_frame.empty:
        return {}

    equity = equity_frame["equity"]
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan)

    out = {
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": (
            float(equity.iloc[-1] / equity.iloc[0] - 1.0) if equity.iloc[0] != 0 else np.nan
        ),
        "max_drawdown": compute_max_drawdown(equity),
        "sharpe": compute_sharpe(returns),
    }

    if not trades_frame.empty and "pnl" in trades_frame.columns:
        out["num_trades"] = float(len(trades_frame))
        out["win_rate"] = compute_win_rate(trades_frame["pnl"])
        out["profit_factor"] = compute_profit_factor(trades_frame["pnl"])
        out["avg_trade_pnl"] = float(trades_frame["pnl"].mean())
        out["avg_trade_pnl_pct"] = (
            float(trades_frame["pnl_pct"].mean()) if "pnl_pct" in trades_frame.columns else np.nan
        )
        out["expectancy"] = compute_expectancy(trades_frame["pnl"])

    return out


def summarize_by_regime(
    trades_frame: pd.DataFrame,
    regime_col: str = "regime_key",
) -> dict[str, dict[str, float]]:
    """
    Per-regime performance breakdown.

    Returns a dict keyed by regime_key. Each value contains:
        win_rate, avg_return, trade_count, expectancy,
        profit_factor, sharpe (annualised, 252-trade basis).

    Use this to identify regimes with negative expectancy and gate them off.
    """
    if trades_frame.empty or regime_col not in trades_frame.columns:
        return {}

    result: dict[str, dict[str, float]] = {}
    pnl_col = "pnl" if "pnl" in trades_frame.columns else None
    pnl_pct_col = "pnl_pct" if "pnl_pct" in trades_frame.columns else None

    if pnl_col is None:
        return {}

    for regime, group in trades_frame.groupby(regime_col):
        pnl = group[pnl_col]
        stats: dict[str, float] = {
            "trade_count": float(len(pnl)),
            "win_rate": compute_win_rate(pnl),
            "avg_return": float(pnl.mean()),
            "expectancy": compute_expectancy(pnl),
            "profit_factor": compute_profit_factor(pnl),
            "max_drawdown": (
                float(pnl.cumsum().sub(pnl.cumsum().cummax()).min()) if len(pnl) > 1 else np.nan
            ),
        }
        if pnl_pct_col is not None:
            pnl_pct = group[pnl_pct_col]
            # Treat each trade-pct as one "period" for a rough per-trade Sharpe.
            stats["sharpe"] = compute_sharpe(pnl_pct, periods_per_year=252)
            stats["avg_return_pct"] = float(pnl_pct.mean())
        result[str(regime)] = stats

    return result


def gate_regimes_by_expectancy(
    regime_stats: dict[str, dict[str, float]],
    expectancy_floor: float = 0.0,
) -> set[str]:
    """
    Return the set of regime keys whose expectancy is below the floor.

    Callers should skip trades when the current regime_key is in this set:

        bad_regimes = gate_regimes_by_expectancy(summarize_by_regime(trades))
        if regime_key in bad_regimes:
            skip_trade()
    """
    return {
        regime
        for regime, stats in regime_stats.items()
        if np.isfinite(stats.get("expectancy", np.nan)) and stats["expectancy"] < expectancy_floor
    }
