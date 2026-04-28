from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from rlm.agents.gate import SystemGate
from rlm.features.scoring.coordinate_regime import classify_regime_from_coordinates
from rlm.roee.coordinate_strategy_router import select_strategy_from_coordinates
from rlm.roee.policy import select_trade, select_trade_from_strategy_name
from rlm.roee.regime_persistence import (
    RegimePersistenceState,
    persistence_size_multiplier,
    persistence_trade_gate,
)
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
_COORDINATE_CONFIDENCE_COLUMNS = (
    "M_alignment",
    "M_trend_strength",
    "M_delta_neutral",
    "M_R_trans",
)
_COORDINATE_LOG_FIELDS = (
    "M_D",
    "M_V",
    "M_L",
    "M_G",
    "M_alignment",
    "M_trend_strength",
    "M_delta_neutral",
    "M_R_trans",
    "M_regime",
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


def _extract_regime_probabilities(row: pd.Series) -> tuple[np.ndarray | None, str]:
    for col, model_name in (("hmm_probs", "hmm"), ("markov_probs", "markov")):
        if col not in row or row.get(col) is None:
            continue
        probs = np.array(row[col], dtype=float)
        if probs.size == 0 or not np.isfinite(probs).all():
            continue
        return probs, model_name
    return None, "none"


def compute_regime_modulators(
    row: pd.Series,
    confidence_threshold: float,
    sizing_multiplier: float,
    transition_penalty: float,
    kronos_confidence_weight: float = 0.4,
    hmm_confidence_weight: float = 0.6,
    kronos_transition_penalty: float = 0.3,
) -> dict[str, float | bool | str]:
    """
    Compute a composite regime confidence and derive gating/sizing modulators for trading.

    Parameters:
        row (pd.Series): Input data row containing regime probabilities and optional Kronos fields.
        confidence_threshold (float): Minimum composite confidence required to allow a trade.
        sizing_multiplier (float): Base multiplier applied when computing the size factor.
        transition_penalty (float): Penalty applied to sizing proportional
            to transition risk (1 - confidence).
        kronos_confidence_weight (float): Weight applied to Kronos agreement
            when blending with HMM/Markov confidence.
        hmm_confidence_weight (float): Weight applied to HMM/Markov confidence
            when blending with Kronos agreement.
        kronos_transition_penalty (float): Additional multiplicative penalty
            applied to composite confidence when a Kronos transition flag
            is present.

    Returns:
        dict[str, float | bool | str]: A dictionary with:
            - "confidence": composite confidence used for gating and sizing (float).
            - "size_mult": computed size multiplier (float, >= 0.0).
            - "trade": `true` if composite confidence >= confidence_threshold, `false` otherwise.
            - "model": source label for the confidence ("hmm", "markov",
              "kronos", or appended with "+kronos").
    """
    probs, model_name = _extract_regime_probabilities(row)

    # --- HMM / Markov baseline confidence ---
    if probs is not None:
        max_prob = float(probs.max())
        hmm_confidence = max_prob
    else:
        hmm_confidence = None

    # --- Kronos confidence (if present) ---
    kronos_agree = _finite_float(row.get("kronos_regime_agreement"), default=np.nan)
    kronos_agree = kronos_agree if math.isfinite(kronos_agree) else None
    kronos_trans = bool(row.get("kronos_transition_flag", False))

    # --- Blend into composite confidence ---
    if hmm_confidence is not None and kronos_agree is not None:
        composite = hmm_confidence_weight * hmm_confidence + kronos_confidence_weight * kronos_agree
        model_name = f"{model_name}+kronos"
    elif hmm_confidence is not None:
        composite = hmm_confidence
    elif kronos_agree is not None:
        composite = kronos_agree
        model_name = "kronos"
    else:
        return {"confidence": 0.0, "size_mult": 0.0, "trade": False, "model": "none"}

    if kronos_trans:
        composite *= 1.0 - kronos_transition_penalty

    trans_risk = 1.0 - composite
    size_mult = sizing_multiplier * composite * (1.0 - transition_penalty * trans_risk)
    trade = composite >= confidence_threshold
    return {
        "confidence": float(composite),
        "size_mult": max(float(size_mult), 0.0),
        "trade": trade,
        "model": model_name,
    }


def compute_hmm_modulators(
    row: pd.Series,
    hmm_confidence_threshold: float,
    sizing_multiplier: float,
    transition_penalty: float,
) -> dict[str, float | bool | str]:
    return compute_regime_modulators(
        row,
        confidence_threshold=hmm_confidence_threshold,
        sizing_multiplier=sizing_multiplier,
        transition_penalty=transition_penalty,
    )


def resolve_latent_regime_from_row(row: pd.Series) -> dict[str, str | float | None]:
    if "markov_state_label" in row.index and pd.notna(row.get("markov_state_label")):
        confidence = _finite_float(row.get("markov_confidence"), default=np.nan)
        return {
            "label": str(row["markov_state_label"]),
            "confidence": confidence if math.isfinite(confidence) else None,
            "source": "markov",
        }

    if "hmm_state_label" in row.index and pd.notna(row.get("hmm_state_label")):
        confidence = None
        if "hmm_probs" in row.index and row.get("hmm_probs") is not None:
            probs = np.array(row["hmm_probs"], dtype=float)
            if probs.size > 0 and np.isfinite(probs).all():
                confidence = float(probs.max())
        if confidence is None and "hmm_confidence" in row.index and pd.notna(row.get("hmm_confidence")):
            hmm_confidence = _finite_float(row.get("hmm_confidence"), default=np.nan)
            confidence = hmm_confidence if math.isfinite(hmm_confidence) else None
        return {
            "label": str(row["hmm_state_label"]),
            "confidence": confidence,
            "source": "hmm",
        }

    return {"label": None, "confidence": None, "source": None}


def _has_coordinate_columns(row: pd.Series) -> bool:
    return all(col in row.index and pd.notna(row.get(col)) for col in _COORDINATE_CONFIDENCE_COLUMNS)


def _resolve_coordinate_regime(row: pd.Series) -> str:
    if "M_regime" in row.index and pd.notna(row.get("M_regime")):
        return str(row["M_regime"])
    return classify_regime_from_coordinates(row)


def _has_coordinate_router_inputs(row: pd.Series) -> bool:
    required = ("M_D", "M_V", "M_L", "M_G", "M_delta_neutral", "M_R_trans")
    return all(col in row.index and pd.notna(row.get(col)) for col in required)


def _model_health_confidence_multiplier(row: pd.Series) -> float:
    model_health = row.get("model_health", None)
    if not isinstance(model_health, dict):
        return 1.0
    if bool(model_health.get("is_stale", False)):
        return 0.5
    if _finite_float(model_health.get("feature_drift_score", 0.0), 0.0) > 1.0:
        return 0.75
    return 1.0


def select_trade_for_row(
    row: pd.Series,
    *,
    strike_increment: float,
    hmm_confidence_threshold: float | None = None,
    hmm_sizing_multiplier: float = 1.0,
    hmm_transition_penalty: float = 0.5,
    short_dte: bool = False,
    use_dynamic_sizing: bool = False,
    vol_target: float = 0.15,
    max_kelly_fraction: float = 0.25,
    max_capital_fraction: float = 0.5,
    regime_adjusted_kelly: bool = True,
    high_vol_kelly_multiplier: float = 0.5,
    transition_kelly_multiplier: float = 0.75,
    calm_trend_kelly_multiplier: float = 1.25,
    vault_uncertainty_threshold: float | None = 0.03,
    vault_size_multiplier: float = 0.5,
    regime_train_sample_count: int | None = None,
    min_regime_train_samples: int | None = None,
    regime_purge_bars: int = 0,
    use_persistence_controls: bool = False,
    use_volume_profile_gating: bool = False,
    wyckoff_threshold: float = 0.7,
    balance_haircut: float = 0.5,
    eighty_percent_boost: float = 0.2,
    hybrid_strength_scaling: bool = True,
    gex_confluence_enabled: bool = True,
    gex_confluence_poc: float | None = None,
    gate: Optional[SystemGate] = None,
) -> TradeDecision:
    """
    Single-bar ROEE decision for backtests and batch pipelines.

    When ``hmm_confidence_threshold`` is None, HMM columns are ignored
    (same as :func:`select_trade`).
    When set, rows with ``hmm_probs`` are gated and size is scaled like :func:`apply_roee_policy`.

    Parameters
    ----------
    short_dte:
        Forward to :func:`select_trade` to activate 0DTE / 1DTE intraday strategy selection.
    """
    if gate is not None and not gate.is_trading_allowed():
        return TradeDecision(
            action="skip",
            strategy_name="system_gate_block",
            regime_key=str(row.get("regime_key", "")),
            rationale=f"System gate is {gate.load().posture} / {gate.load().status}. Trading paused.",
            metadata={"gate_status": gate.load().status, "gate_posture": gate.load().posture},
        )

    missing = [c for c in _SELECT_TRADE_ROW_COLUMNS if c not in row.index]
    if missing:
        return TradeDecision(
            action="skip",
            rationale=f"Missing required row columns: {missing}",
            metadata={"missing_columns": missing},
        )

    use_hmm = hmm_confidence_threshold is not None
    min_regime_samples = max(int(min_regime_train_samples), 0) if min_regime_train_samples is not None else 0
    train_sample_count = max(int(regime_train_sample_count), 0) if regime_train_sample_count is not None else 0

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

    latent_regime = resolve_latent_regime_from_row(row)
    coordinate_router_strategy: str | None = None
    coordinate_regime: str | None = None
    strategy_source = "legacy_regime_map"

    if _has_coordinate_router_inputs(row):
        regime_row = row.to_dict()
        regime_row["M_regime"] = _resolve_coordinate_regime(row)
        coordinate_regime = str(regime_row["M_regime"])
        coordinate_router_strategy = select_strategy_from_coordinates(regime_row)
        strategy_source = "coordinate_router"
        if coordinate_router_strategy == "no_trade":
            return TradeDecision(
                action="skip",
                strategy_name="coordinate_router_no_trade",
                regime_key=str(row.get("regime_key", "")),
                rationale="Coordinate regime filter blocked trade.",
                metadata={
                    "strategy_source": strategy_source,
                    "coordinate_strategy": coordinate_router_strategy,
                    "M_regime": regime_row["M_regime"],
                },
            )

    persistence_size_mult = 1.0
    enable_persistence_controls = use_persistence_controls or bool(row.get("use_persistence_controls", False))
    if enable_persistence_controls:
        p_state = RegimePersistenceState(
            regime=str(row.get("M_regime", "unknown")),
            consecutive_bars=int(row.get("M_regime_consecutive", 1)),
            dominant_probability=float(row.get("M_regime_prob", 0.5)),
            flip_rate_recent=float(row.get("M_regime_flip_rate_recent", 0.0)),
        )
        if not persistence_trade_gate(p_state):
            return TradeDecision(
                action="skip",
                strategy_name="persistence_block",
                regime_key=str(row.get("regime_key", "")),
                rationale="Regime persistence gate blocked trade.",
                metadata={
                    "strategy_source": "persistence_gate",
                    "M_regime": p_state.regime,
                    "M_regime_prob": p_state.dominant_probability,
                    "M_regime_consecutive": p_state.consecutive_bars,
                    "M_regime_flip_rate_recent": p_state.flip_rate_recent,
                },
            )
        persistence_size_mult = persistence_size_multiplier(p_state)

    if use_hmm:
        mod = compute_regime_modulators(
            row,
            confidence_threshold=float(hmm_confidence_threshold),
            sizing_multiplier=hmm_sizing_multiplier,
            transition_penalty=hmm_transition_penalty,
        )
        if not bool(mod["trade"]):
            regime_model = str(mod["model"])
            return TradeDecision(
                action="skip",
                strategy_name=f"{regime_model}_gate" if regime_model != "none" else "regime_gate",
                regime_key=str(row.get("regime_key", "")),
                rationale=(
                    f"{regime_model.upper()} confidence below threshold"
                    if regime_model != "none"
                    else "Regime confidence below threshold"
                ),
                metadata={
                    "regime_model": regime_model,
                    "regime_confidence": mod["confidence"],
                    "regime_size_mult": mod["size_mult"],
                    "regime_trade_allowed": False,
                    "hmm_confidence": mod["confidence"],
                    "hmm_size_mult": mod["size_mult"],
                    "hmm_trade_allowed": False,
                },
            )

    trade_kwargs = dict(
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
            _finite_float(row.get("realized_vol"), default=np.nan) if pd.notna(row.get("realized_vol")) else None
        ),
        use_dynamic_sizing=use_dynamic_sizing,
        vol_target=vol_target,
        max_kelly_fraction=max_kelly_fraction,
        max_capital_fraction=max_capital_fraction,
        regime_adjusted_kelly=regime_adjusted_kelly,
        regime_state_label=(str(latent_regime["label"]) if latent_regime["label"] is not None else None),
        regime_state_confidence=(
            float(latent_regime["confidence"]) if latent_regime["confidence"] is not None else None
        ),
        high_vol_kelly_multiplier=high_vol_kelly_multiplier,
        transition_kelly_multiplier=transition_kelly_multiplier,
        calm_trend_kelly_multiplier=calm_trend_kelly_multiplier,
        vault_uncertainty_threshold=vault_uncertainty_threshold,
        vault_size_multiplier=vault_size_multiplier,
        short_dte=short_dte,
        use_volume_profile_gating=use_volume_profile_gating,
        effort_result_divergence=(
            _finite_float(row.get("vp_effort_result_score"), default=np.nan)
            if pd.notna(row.get("vp_effort_result_score"))
            else None
        ),
        auction_state=(str(row.get("vp_auction_state")) if pd.notna(row.get("vp_auction_state")) else None),
        eighty_percent_rule_signal=bool(row.get("vp_eighty_percent_signal", False)),
        cumulative_wyckoff_score=(
            _finite_float(row.get("cumulative_wyckoff_score"), default=np.nan)
            if pd.notna(row.get("cumulative_wyckoff_score"))
            else None
        ),
        wyckoff_threshold=wyckoff_threshold,
        balance_haircut=balance_haircut,
        eighty_percent_boost=eighty_percent_boost,
        hybrid_strength_scaling=hybrid_strength_scaling,
        hybrid_strength=(
            _finite_float(row.get("vp_hybrid_strength_max"), default=np.nan)
            if pd.notna(row.get("vp_hybrid_strength_max"))
            else None
        ),
        gex_confluence_enabled=gex_confluence_enabled,
        gex_confluence_poc=(
            _finite_float(row.get("vp_gex_confluence_poc"), default=np.nan)
            if pd.notna(row.get("vp_gex_confluence_poc"))
            else None
        ),
    )
    if coordinate_router_strategy is not None:
        decision = select_trade_from_strategy_name(
            strategy_name=coordinate_router_strategy,
            **trade_kwargs,
        )
    else:
        decision = select_trade(**trade_kwargs)

    if use_hmm and decision.action == "enter":
        mod = compute_regime_modulators(
            row,
            confidence_threshold=float(hmm_confidence_threshold),
            sizing_multiplier=hmm_sizing_multiplier,
            transition_penalty=hmm_transition_penalty,
        )
        base_sf = float(decision.size_fraction or 0.0)
        meta = dict(decision.metadata)
        meta["regime_model"] = str(mod["model"])
        meta["regime_confidence"] = mod["confidence"]
        meta["regime_size_mult"] = mod["size_mult"]
        meta["regime_trade_allowed"] = True
        meta["hmm_confidence"] = mod["confidence"]
        meta["hmm_size_mult"] = mod["size_mult"]
        meta["hmm_trade_allowed"] = True
        meta["persistence_size_mult"] = persistence_size_mult
        meta["persistence_controls_enabled"] = enable_persistence_controls
        meta["strategy_source"] = strategy_source
        if coordinate_router_strategy is not None:
            meta["coordinate_strategy"] = coordinate_router_strategy
        if coordinate_regime is not None:
            meta["M_regime"] = coordinate_regime
        for field in _COORDINATE_LOG_FIELDS:
            if field in row.index and pd.notna(row.get(field)):
                meta[field] = row[field]
        if _has_coordinate_columns(row):
            coord_conf = 1.0
            coord_conf *= 1.0 + (_finite_float(row.get("M_alignment"), 0.0) / 25.0)
            coord_conf *= 1.0 + (_finite_float(row.get("M_trend_strength"), 0.0) / 5.0)
            coord_conf *= 1.0 - (_finite_float(row.get("M_delta_neutral"), 0.0) / 10.0)
            if _finite_float(row.get("M_R_trans"), 0.0) > 5.0:
                coord_conf *= 0.5
            coord_conf = max(coord_conf, 0.0)
            meta["coordinate_confidence_multiplier"] = float(coord_conf)
            base_sf = base_sf * float(coord_conf)
        health_mult = _model_health_confidence_multiplier(row)
        meta["model_health_confidence_multiplier"] = float(health_mult)
        if latent_regime["source"] is not None:
            meta["kelly_latent_regime_source"] = str(latent_regime["source"])
        return replace(
            decision,
            size_fraction=quantize_fraction(
                base_sf * float(mod["size_mult"]) * float(persistence_size_mult) * float(health_mult)
            ),
            metadata=meta,
        )

    if use_hmm:
        mod = compute_regime_modulators(
            row,
            confidence_threshold=float(hmm_confidence_threshold),
            sizing_multiplier=hmm_sizing_multiplier,
            transition_penalty=hmm_transition_penalty,
        )
        meta = dict(decision.metadata)
        meta["regime_model"] = str(mod["model"])
        meta["regime_confidence"] = mod["confidence"]
        meta["regime_size_mult"] = mod["size_mult"]
        meta["regime_trade_allowed"] = True
        meta["hmm_confidence"] = mod["confidence"]
        meta["hmm_size_mult"] = mod["size_mult"]
        meta["hmm_trade_allowed"] = True
        meta["persistence_size_mult"] = persistence_size_mult
        meta["persistence_controls_enabled"] = enable_persistence_controls
        meta["strategy_source"] = strategy_source
        if coordinate_router_strategy is not None:
            meta["coordinate_strategy"] = coordinate_router_strategy
        if coordinate_regime is not None:
            meta["M_regime"] = coordinate_regime
        for field in _COORDINATE_LOG_FIELDS:
            if field in row.index and pd.notna(row.get(field)):
                meta[field] = row[field]
        if latent_regime["source"] is not None:
            meta["kelly_latent_regime_source"] = str(latent_regime["source"])
        return replace(decision, metadata=meta)

    if decision.action == "enter":
        meta = dict(decision.metadata)
        health_mult = _model_health_confidence_multiplier(row)
        meta["model_health_confidence_multiplier"] = float(health_mult)
        if latent_regime["source"] is not None:
            meta["kelly_latent_regime_source"] = str(latent_regime["source"])
        meta["strategy_source"] = strategy_source
        if coordinate_router_strategy is not None:
            meta["coordinate_strategy"] = coordinate_router_strategy
        if coordinate_regime is not None:
            meta["M_regime"] = coordinate_regime
        for field in _COORDINATE_LOG_FIELDS:
            if field in row.index and pd.notna(row.get(field)):
                meta[field] = row[field]
        if _has_coordinate_columns(row):
            coord_conf = 1.0
            coord_conf *= 1.0 + (_finite_float(row.get("M_alignment"), 0.0) / 25.0)
            coord_conf *= 1.0 + (_finite_float(row.get("M_trend_strength"), 0.0) / 5.0)
            coord_conf *= 1.0 - (_finite_float(row.get("M_delta_neutral"), 0.0) / 10.0)
            if _finite_float(row.get("M_R_trans"), 0.0) > 5.0:
                coord_conf *= 0.5
            coord_conf = max(coord_conf, 0.0)
            meta["coordinate_confidence_multiplier"] = float(coord_conf)
            return replace(
                decision,
                size_fraction=quantize_fraction(
                    float(decision.size_fraction or 0.0)
                    * float(coord_conf)
                    * float(persistence_size_mult)
                    * float(health_mult)
                ),
                metadata=meta,
            )
        return replace(
            decision,
            size_fraction=quantize_fraction(
                float(decision.size_fraction or 0.0) * float(persistence_size_mult) * float(health_mult)
            ),
            metadata=meta,
        )

    return decision
