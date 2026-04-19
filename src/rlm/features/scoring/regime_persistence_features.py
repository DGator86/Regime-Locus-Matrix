from __future__ import annotations

import pandas as pd


def add_regime_persistence_columns(
    df: pd.DataFrame,
    regime_col: str = "M_regime",
    prob_col: str = "M_regime_prob",
    lookback: int = 10,
) -> pd.DataFrame:
    out = df.copy()
    consecutive = []
    flip_rate_recent = []
    prev = None
    streak = 0
    history: list[str] = []

    for regime in out[regime_col].astype(str).tolist():
        if regime == prev:
            streak += 1
        else:
            streak = 1
            prev = regime
        consecutive.append(streak)

        history.append(regime)
        recent = history[-lookback:]
        flips = sum(recent[i] != recent[i - 1] for i in range(1, len(recent)))
        flip_rate_recent.append(flips / max(len(recent) - 1, 1))

    out["M_regime_consecutive"] = consecutive
    out["M_regime_flip_rate_recent"] = flip_rate_recent
    if prob_col not in out.columns:
        out["M_regime_prob"] = 0.5
    return out
