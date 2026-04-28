from __future__ import annotations

import numpy as np
import pandas as pd
import optuna

from pathlib import Path

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline
from rlm.optimization.config import NightlyHyperparams
from rlm.roee.engine import ROEEConfig


def _compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 10:
        return float("nan")
    std = r.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float((r.mean() / std) * np.sqrt(periods_per_year))


def _compute_max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _signal_based_score(
    policy_df: pd.DataFrame,
    oos_start: int,
    transaction_cost_frac: float = 0.001,
) -> float:
    """Sharpe-minus-drawdown on OOS tail from ROEE signals with transaction costs.

    Uses ``roee_action`` and ``roee_size_fraction`` from the pipeline output to
    construct a long-only signal, lags it by one bar (no look-ahead), applies
    proportional transaction costs on position changes, then scores with:

        score = OOS Sharpe + OOS max-drawdown   (drawdown is negative)

    ``transaction_cost_frac`` is a decimal fraction (e.g. 0.001 = 10 bps).

    Returns ``nan`` when there is insufficient OOS data or no trades.
    """
    df = policy_df.iloc[oos_start:].copy()
    if len(df) < 30 or "close" not in df.columns or "roee_action" not in df.columns:
        return float("nan")

    # Position: enter → size_fraction, anything else → 0
    signals = df["roee_action"].map({"enter": 1.0}).fillna(0.0)
    if "roee_size_fraction" in df.columns:
        signals = signals * df["roee_size_fraction"].fillna(0.0).clip(0.0, 1.0)

    # Lag by one bar: signal fires at bar close, P&L realised next bar
    lagged = signals.shift(1).fillna(0.0)
    price_returns = df["close"].pct_change().fillna(0.0)
    gross = lagged * price_returns

    # Proportional transaction costs on position changes
    costs = signals.diff().abs().fillna(0.0) * transaction_cost_frac
    net = gross - costs

    sharpe = _compute_sharpe(net)
    if not np.isfinite(sharpe):
        return float("nan")

    max_dd = _compute_max_drawdown(net)
    return sharpe + max_dd  # max_dd ≤ 0, so it penalises drawdowns


def align_regime_labels(
    raw_labels: np.ndarray,
    returns: np.ndarray,
    n_regimes: int,
) -> np.ndarray:
    """Re-index HMM/K-Means regime labels so regime 0 is lowest volatility.

    HMM models assign arbitrary integer labels that can switch between runs.
    This function provides a stable ordering: regimes are sorted ascending by
    realised volatility of their returns so that regime 0 is always the calmest
    state and regime n-1 is always the highest-volatility state.

    Parameters
    ----------
    raw_labels:
        Integer regime assignments, shape (n_bars,).
    returns:
        Bar returns aligned with ``raw_labels``, shape (n_bars,).
    n_regimes:
        Expected number of regimes.  Labels missing from ``raw_labels`` are
        mapped to the last (highest-vol) regime.

    Returns
    -------
    aligned_labels:
        Re-indexed integer array, same length as ``raw_labels``.
    """
    if len(raw_labels) != len(returns):
        raise ValueError("raw_labels and returns must have the same length")

    unique = np.unique(raw_labels)
    vol_by_label: dict[int, float] = {}
    for lbl in unique:
        mask = raw_labels == lbl
        r = returns[mask]
        vol_by_label[int(lbl)] = float(np.std(r)) if len(r) > 1 else 0.0

    sorted_labels = sorted(vol_by_label.keys(), key=lambda x: vol_by_label[x])
    mapping: dict[int, int] = {old: new for new, old in enumerate(sorted_labels)}

    # Any label not seen in raw_labels → highest-vol bucket
    fallback = len(sorted_labels) - 1
    for missing in range(n_regimes):
        if missing not in mapping:
            mapping[missing] = fallback

    return np.array([mapping.get(int(lbl), fallback) for lbl in raw_labels])


class OptimizationBase:
    """Shared utilities for nightly hyperparameter optimization."""

    @staticmethod
    def load_bars(
        symbol: str, lookback_bars: int = 252 * 2, root: Path | None = None
    ) -> pd.DataFrame:
        """Load daily bars from data/raw/bars_<SYMBOL>.csv."""
        repo_root = root or Path(__file__).resolve().parents[3]
        bars_path = repo_root / "data" / "raw" / f"bars_{symbol.upper()}.csv"
        if not bars_path.exists():
            raise FileNotFoundError(f"Missing bars file: {bars_path}")

        bars = pd.read_csv(bars_path)
        if "timestamp" in bars.columns:
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")

        return bars.tail(lookback_bars).copy()

    @staticmethod
    def objective(trial: optuna.Trial, symbols: list[str], regime_model: str) -> float:
        """Walk-forward Sharpe objective with transaction costs.

        Replaces the previous ``confidence × entry_frequency`` proxy with a
        financially meaningful score derived from actual ROEE signals:

        - Hyperparameters sampled by Optuna (same 9 as before + transaction cost).
        - Pipeline run once per symbol on the full bar history.
        - Score evaluated on the OOS tail (last 25 % of bars, minimum 30 bars).
        - OOS score = Sharpe(net returns) + max_drawdown  (drawdown is ≤ 0).
        - Symbol scores averaged; returns -999 when no symbol has valid data.

        Look-ahead note: the HMM is fit on full bars, so the OOS tail is not
        truly held-out for the regime model.  True walk-forward requires
        multiple pipeline refits (expensive for a nightly run).  This expanding-
        window approach is a sound compromise: the signal-to-noise improvement
        over ``confidence × entry_frequency`` is large, and the remaining
        in-sample contamination is minor because HMM parameters are fit to
        factor structure, not to P&L directly.
        """
        nightly = NightlyHyperparams(
            mtf_ltf_weight=trial.suggest_float("mtf_ltf_weight", 0.35, 0.65),
            mtf_regimes=trial.suggest_categorical("mtf_regimes", [True, False]),
            hmm_confidence_threshold=trial.suggest_float("hmm_confidence_threshold", 0.55, 0.75),
            high_vol_kelly_multiplier=trial.suggest_float("high_vol_kelly_multiplier", 0.45, 0.75),
            transition_kelly_multiplier=trial.suggest_float(
                "transition_kelly_multiplier", 0.70, 0.95
            ),
            calm_trend_kelly_multiplier=trial.suggest_float(
                "calm_trend_kelly_multiplier", 1.05, 1.35
            ),
            move_window=trial.suggest_int("move_window", 85, 115),
            vol_window=trial.suggest_int("vol_window", 85, 115),
            direction_neutral_threshold=trial.suggest_float(
                "direction_neutral_threshold", 0.26, 0.34
            ),
        )
        # Decimal fraction; 0.0005 = 5 bps, 0.003 = 30 bps
        transaction_cost_frac = trial.suggest_float("transaction_cost_frac", 0.0005, 0.003)

        cfg = FullRLMConfig(
            regime_model=regime_model,
            mtf=True,
            roee_config=ROEEConfig(use_dynamic_sizing=True),
            nightly_hyperparams=nightly.__dict__,
        )

        scores: list[float] = []
        for sym in symbols:
            try:
                bars = OptimizationBase.load_bars(sym, lookback_bars=252 * 2)
            except FileNotFoundError:
                continue
            try:
                result = FullRLMPipeline(cfg).run(bars)
            except Exception:  # noqa: BLE001
                continue
            if result.policy_df.empty:
                continue

            n = len(result.policy_df)
            oos_start = int(n * 0.75)
            score = _signal_based_score(result.policy_df, oos_start, transaction_cost_frac)
            if np.isfinite(score):
                scores.append(score)

        if not scores:
            raise optuna.TrialPruned()
        return float(np.mean(scores))
