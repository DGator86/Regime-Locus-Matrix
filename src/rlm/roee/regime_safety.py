from __future__ import annotations

import pandas as pd


def build_regime_safety_rationale(
    *,
    regime_key: str,
    regime_train_sample_count: int,
    min_regime_train_samples: int,
    purge_bars: int,
) -> str:
    return (
        "Regime safety check failed: "
        f"{regime_train_sample_count} prior training samples for {regime_key!r} "
        f"with purge_bars={purge_bars} < required {min_regime_train_samples}."
    )


def attach_regime_safety_columns(
    df: pd.DataFrame,
    *,
    min_regime_train_samples: int,
    purge_bars: int = 0,
) -> pd.DataFrame:
    if "regime_key" not in df.columns:
        raise ValueError("Missing required column for regime safety: ['regime_key']")

    out = df.copy()
    min_samples = max(int(min_regime_train_samples), 0)
    purge = max(int(purge_bars), 0)
    regime_keys = out["regime_key"].fillna("").astype(str)

    if regime_keys.empty:
        out["regime_train_sample_count"] = pd.Series(dtype="int64")
        out["regime_train_sample_requirement"] = pd.Series(dtype="int64")
        out["regime_train_sample_gap"] = pd.Series(dtype="int64")
        out["regime_train_purge_bars"] = pd.Series(dtype="int64")
        out["regime_safety_ok"] = pd.Series(dtype="bool")
        return out

    dummy_counts = pd.get_dummies(regime_keys)
    prior_counts = dummy_counts.cumsum().shift(purge + 1, fill_value=0)
    sample_counts = [
        int(prior_counts.iloc[i].get(regime_keys.iat[i], 0))
        if regime_keys.iat[i]
        else 0
        for i in range(len(out))
    ]

    count_series = pd.Series(sample_counts, index=out.index, dtype="int64")
    out["regime_train_sample_count"] = count_series
    out["regime_train_sample_requirement"] = min_samples
    out["regime_train_sample_gap"] = (min_samples - count_series).clip(lower=0).astype("int64")
    out["regime_train_purge_bars"] = purge
    out["regime_safety_ok"] = count_series >= min_samples
    return out
