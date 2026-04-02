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
        "total_return_pct": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if equity.iloc[0] != 0 else np.nan,
        "max_drawdown": compute_max_drawdown(equity),
        "sharpe": compute_sharpe(returns),
    }

    if not trades_frame.empty and "pnl" in trades_frame.columns:
        out["num_trades"] = float(len(trades_frame))
        out["win_rate"] = compute_win_rate(trades_frame["pnl"])
        out["profit_factor"] = compute_profit_factor(trades_frame["pnl"])
        out["avg_trade_pnl"] = float(trades_frame["pnl"].mean())
        out["avg_trade_pnl_pct"] = float(trades_frame["pnl_pct"].mean()) if "pnl_pct" in trades_frame.columns else np.nan

    return out
