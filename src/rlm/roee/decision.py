from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from rlm.roee.policy import select_trade
from rlm.roee.regime_safety import build_regime_safety_rationale
from rlm.roee.sizing import quantize_fraction
from rlm.types.options import TradeDecision

_SELECT_TRADE_ROW_COLUMNS = (
    "close",
    "sigma",
    "S_D",
    "S_V",
    "S_L",
    "S_G",
    "direction_regime",
    "volatility_regime",
    "liquidity_regime",
    "dealer_flow_regime",
    "regime_key",
)


def _finite_float(x: object, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except TypeError:
        pass
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def compute_hmm_modulators(
    row: pd.Series,
    hmm_confidence_threshold: float,
    sizing_multiplier: float,
    transition_penalty: float,
) -> dict[str, float | bool]:
    if "hmm_probs" not in row or row.get("hmm_probs") is None:
        return {"confidence": 1.0, "size_mult": 1.0, "trade": True}

    probs = np.array(row["hmm_probs"], dtype=float)
    if probs.size == 0 or not np.isfinite(probs).all():
        return {"confidence": 1.0, "size_mult": 1.0, "trade": True}

    max_prob = float(probs.max())
    trans_risk = 1.0 - max_prob
    confidence = max_prob
    size_mult = sizing_multiplier * max_prob * (1.0 - transition_penalty * trans_risk)
    trade = confidence >= hmm_confidence_threshold
    return {"confidence": confidence, "size_mult": max(float(size_mult), 0.0), "trade": trade}


def select_trade_for_row(
    row: pd.Series,
    *,
    strike_increment: float,
    hmm_confidence_threshold: float | None = None,
    hmm_sizing_multiplier: float = 1.0,
    hmm_transition_penalty: float = 0.5,
    use_dynamic_sizing: bool = False,
    vol_target: float = 0.15,
    max_kelly_fraction: float = 0.25,
    max_capital_fraction: float = 0.5,
    vault_uncertainty_threshold: float | None = 0.03,
    vault_size_multiplier: float = 0.5,
    regime_train_sample_count: int | None = None,
    min_regime_train_samples: int | None = None,
    regime_purge_bars: int = 0,
) -> TradeDecision:
    """
    Single-bar ROEE decision for backtests and batch pipelines.

    When ``hmm_confidence_threshold`` is None, HMM columns are ignored (same as :func:`select_trade`).
    When set, rows with ``hmm_probs`` are gated and size is scaled like :func:`apply_roee_policy`.
    """
    missing = [c for c in _SELECT_TRADE_ROW_COLUMNS if c not in row.index]
    if missing:
        return TradeDecision(
            action="skip",
            rationale=f"Missing required row columns: {missing}",
            metadata={"missing_columns": missing},
        )

    use_hmm = hmm_confidence_threshold is not None
    min_regime_samples = (
        max(int(min_regime_train_samples), 0) if min_regime_train_samples is not None else 0
    )
    train_sample_count = (
        max(int(regime_train_sample_count), 0) if regime_train_sample_count is not None else 0
    )

    if min_regime_samples > 0 and train_sample_count < min_regime_samples:
        return TradeDecision(
            action="hold",
            strategy_name="regime_safety_check",
            regime_key=str(row.get("regime_key", "")),
            rationale=build_regime_safety_rationale(
                regime_key=str(row.get("regime_key", "")),
                regime_train_sample_count=train_sample_count,
                min_regime_train_samples=min_regime_samples,
                purge_bars=regime_purge_bars,
            ),
            metadata={
                "regime_train_sample_count": train_sample_count,
                "min_regime_train_samples": min_regime_samples,
                "regime_train_purge_bars": max(int(regime_purge_bars), 0),
                "regime_safety_ok": False,
            },
        )

    if use_hmm:
        mod = compute_hmm_modulators(
            row,
            hmm_confidence_threshold=float(hmm_confidence_threshold),
            sizing_multiplier=hmm_sizing_multiplier,
            transition_penalty=hmm_transition_penalty,
        )
        if not bool(mod["trade"]):
            return TradeDecision(
                action="skip",
                strategy_name="hmm_gate",
                regime_key=str(row.get("regime_key", "")),
                rationale="HMM confidence below threshold",
                metadata={
                    "hmm_confidence": mod["confidence"],
                    "hmm_size_mult": mod["size_mult"],
                    "hmm_trade_allowed": False,
                },
            )

    decision = select_trade(
        current_price=float(row["close"]),
        sigma=float(row["sigma"]),
        s_d=_finite_float(row["S_D"], 0.0),
        s_v=_finite_float(row["S_V"], 0.0),
        s_l=_finite_float(row["S_L"], 0.0),
        s_g=_finite_float(row["S_G"], 0.0),
        direction_regime=str(row["direction_regime"]),
        volatility_regime=str(row["volatility_regime"]),
        liquidity_regime=str(row["liquidity_regime"]),
        dealer_flow_regime=str(row["dealer_flow_regime"]),
        regime_key=str(row["regime_key"]),
        bid_ask_spread_pct=(
            float(row["bid_ask_spread"] / row["close"])
            if "bid_ask_spread" in row.index and pd.notna(row.get("bid_ask_spread"))
            else None
        ),
        has_major_event=(
            bool(row["has_major_event"])
            if "has_major_event" in row.index and pd.notna(row.get("has_major_event"))
            else False
        ),
        strike_increment=strike_increment,
        forecast_return=(
            _finite_float(row.get("forecast_return"), default=np.nan)
            if pd.notna(row.get("forecast_return"))
            else (
                _finite_float(row.get("forecast_return_median"), default=np.nan)
                if pd.notna(row.get("forecast_return_median"))
                else None
            )
        ),
        forecast_uncertainty=(
            _finite_float(row.get("forecast_uncertainty"), default=np.nan)
            if pd.notna(row.get("forecast_uncertainty"))
            else None
        ),
        realized_vol=(
            _finite_float(row.get("realized_vol"), default=np.nan)
            if pd.notna(row.get("realized_vol"))
            else None
        ),
        use_dynamic_sizing=use_dynamic_sizing,
        vol_target=vol_target,
        max_kelly_fraction=max_kelly_fraction,
        max_capital_fraction=max_capital_fraction,
        vault_uncertainty_threshold=vault_uncertainty_threshold,
        vault_size_multiplier=vault_size_multiplier,
    )

    if use_hmm and decision.action == "enter":
        mod = compute_hmm_modulators(
            row,
            hmm_confidence_threshold=float(hmm_confidence_threshold),
            sizing_multiplier=hmm_sizing_multiplier,
            transition_penalty=hmm_transition_penalty,
        )
        base_sf = float(decision.size_fraction or 0.0)
        meta = dict(decision.metadata)
        meta["hmm_confidence"] = mod["confidence"]
        meta["hmm_size_mult"] = mod["size_mult"]
        meta["hmm_trade_allowed"] = True
        return replace(
            decision,
            size_fraction=quantize_fraction(base_sf * float(mod["size_mult"])),
            metadata=meta,
        )

    if use_hmm:
        mod = compute_hmm_modulators(
            row,
            hmm_confidence_threshold=float(hmm_confidence_threshold),
            sizing_multiplier=hmm_sizing_multiplier,
            transition_penalty=hmm_transition_penalty,
        )
        meta = dict(decision.metadata)
        meta["hmm_confidence"] = mod["confidence"]
        meta["hmm_size_mult"] = mod["size_mult"]
        meta["hmm_trade_allowed"] = True
        return replace(decision, metadata=meta)

    return decision