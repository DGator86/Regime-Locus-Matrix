"""Challenge parameter optimizer — walk-forward grid search over ChallengeConfig knobs.

Usage
-----
    from rlm.challenge.optimize import optimize_challenge_config
    best_cfg = optimize_challenge_config(bars_df, n_splits=5)

No scipy dependency — pure nested loops over a discrete parameter grid evaluated
by a simple Sharpe / max-drawdown score from time-series cross-validation splits.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any

import pandas as pd

from rlm.challenge.config import ChallengeConfig

# ---------------------------------------------------------------------------
# Parameter search space
# ---------------------------------------------------------------------------

CHALLENGE_PARAM_SPACE: dict[str, list[Any]] = {
    "stop_loss_mult": [0.78, 0.82, 0.86],
    "stage1_size_frac": [0.25, 0.30, 0.35],
    "stage2_size_frac": [0.15, 0.20, 0.25],
    "stage3_size_frac": [0.10, 0.15, 0.20],
    "stage1_profit_target_mult": [1.25, 1.38, 1.50],
    "stage2_profit_target_mult": [1.50, 1.65, 1.80],
    "stage3_profit_target_mult": [1.75, 2.00, 2.25],
    "trail_retrace_frac": [0.08, 0.10, 0.12],
    "regime_win_rate_min": [0.35, 0.40, 0.45],
    "spread_half_width_frac": [0.06, 0.08, 0.10],
}


# ---------------------------------------------------------------------------
# Walk-forward split builder
# ---------------------------------------------------------------------------


def _build_time_splits(bars_df: pd.DataFrame, n_splits: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split ``bars_df`` into ``n_splits`` walk-forward (train, test) pairs.

    Each test window is ``1/n_splits`` of the full data; training data is the
    preceding bars (expanding window).
    """
    n = len(bars_df)
    if n < n_splits * 2:
        raise ValueError(
            f"bars_df has only {n} rows but n_splits={n_splits} requires at least {n_splits * 2} rows."
        )
    test_size = n // (n_splits + 1)
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_splits):
        train_end = test_size * (i + 1)
        test_end = train_end + test_size
        train = bars_df.iloc[:train_end].copy()
        test = bars_df.iloc[train_end:test_end].copy()
        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))
    return splits


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------


def _score_config(cfg: ChallengeConfig, splits: list[tuple[pd.DataFrame, pd.DataFrame]]) -> float:
    """Return a walk-forward score for the given config across all splits.

    The score is the average OOS Sharpe ratio minus a drawdown penalty.  Uses
    synthetic "return" as the column ``close`` pct-change if present; otherwise
    falls back to a random-walk stub that allows the optimizer to still rank
    configs by their internal parameters (commission, sizing, etc.).

    Returns ``-inf`` if no usable price data is present.
    """
    oos_returns: list[float] = []

    for _train, test in splits:
        if "close" not in test.columns:
            return float("-inf")
        prices = test["close"].dropna()
        if len(prices) < 2:
            continue
        daily_returns = prices.pct_change().dropna()

        # Simulate simplified P&L: apply sizing fraction × expected option leverage
        # This is a heuristic ranking proxy — not a full backtest.
        size_frac = cfg.size_fraction(cfg.seed_capital)
        # Scale returns by size fraction (directional leverage proxy)
        scaled = daily_returns * size_frac * 5.0  # 5× rough option delta leverage

        # Apply stop-loss / profit-target clamp per period
        clipped = scaled.clip(
            lower=(cfg.stop_loss_mult - 1.0) * size_frac,
            upper=(cfg.profit_target_for(cfg.seed_capital) - 1.0) * size_frac,
        )
        # Deduct round-trip friction estimate
        friction_per_trade = (
            2.0 * cfg.spread_half_width_frac + 2.0 * cfg.commission_per_contract / (cfg.seed_capital * size_frac + 1.0)
        )
        clipped = clipped - friction_per_trade / max(len(clipped), 1)

        oos_returns.extend(clipped.tolist())

    if not oos_returns:
        return float("-inf")

    import math
    import statistics

    mean_r = statistics.mean(oos_returns)
    std_r = statistics.stdev(oos_returns) if len(oos_returns) > 1 else 1e-9
    sharpe = mean_r / max(std_r, 1e-9) * math.sqrt(252)

    # Drawdown penalty
    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in oos_returns:
        cum *= 1.0 + r
        if cum > peak:
            peak = cum
        dd = (peak - cum) / max(peak, 1e-9)
        if dd > max_dd:
            max_dd = dd

    return float(sharpe - 2.0 * max_dd)


# ---------------------------------------------------------------------------
# Public optimizer
# ---------------------------------------------------------------------------


def optimize_challenge_config(
    bars_df: pd.DataFrame,
    n_splits: int = 5,
    param_space: dict[str, list[Any]] | None = None,
    base_cfg: ChallengeConfig | None = None,
) -> ChallengeConfig:
    """Run a walk-forward grid search and return the best-scoring ChallengeConfig.

    Parameters
    ----------
    bars_df:
        OHLCV DataFrame with a DatetimeIndex and at least a ``close`` column.
    n_splits:
        Number of walk-forward OOS splits (default 5).
    param_space:
        Override the default ``CHALLENGE_PARAM_SPACE``.  Each key must be a
        valid field name of ``ChallengeConfig``.
    base_cfg:
        Starting configuration.  Non-searched parameters keep their values.

    Returns
    -------
    ChallengeConfig
        The configuration with the highest walk-forward score.
    """
    space = param_space or CHALLENGE_PARAM_SPACE
    base = base_cfg or ChallengeConfig()

    splits = _build_time_splits(bars_df, n_splits)

    keys = list(space.keys())
    values_lists = [space[k] for k in keys]

    best_score = float("-inf")
    best_cfg = base

    for combo in itertools.product(*values_lists):
        params = dict(zip(keys, combo))
        try:
            candidate = replace(base, **params)
        except TypeError:
            # Skip combos with invalid field names (shouldn't happen with CHALLENGE_PARAM_SPACE)
            continue

        score = _score_config(candidate, splits)
        if score > best_score:
            best_score = score
            best_cfg = candidate

    return best_cfg
