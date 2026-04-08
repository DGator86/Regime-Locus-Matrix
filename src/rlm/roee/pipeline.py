from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlm.roee.decision import _finite_float, compute_hmm_modulators
from rlm.roee.policy import select_trade


@dataclass(frozen=True)
class ROEEConfig:
    hmm_confidence_threshold: float = 0.6
    sizing_multiplier: float = 1.0
    transition_penalty: float = 0.5
    use_dynamic_sizing: bool = False
    vol_target: float = 0.15
    max_kelly_fraction: float = 0.25
    max_capital_fraction: float = 0.5


def _hmm_modulators_for_config(row: pd.Series, config: ROEEConfig) -> dict[str, float | bool]:
    return compute_hmm_modulators(
        row,
        hmm_confidence_threshold=config.hmm_confidence_threshold,
        sizing_multiplier=config.sizing_multiplier,
        transition_penalty=config.transition_penalty,
    )


def apply_roee_policy(
    df: pd.DataFrame,
    strike_increment: float = 1.0,
    config: ROEEConfig | None = None,
) -> pd.DataFrame:
    """
    Applies the ROEE decision policy row-by-row and stores summarized outputs.
    """
    required = [
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
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for ROEE policy: {missing}")

    cfg = config or ROEEConfig()

    out = df.copy()

    actions = []
    strategy_names = []
    rationales = []
    size_fractions = []
    target_profit_pcts = []
    max_risk_pcts = []
    leg_counts = []
    hmm_confidences = []
    hmm_size_multipliers = []
    hmm_trade_flags = []

    for _, row in out.iterrows():
        mod = _hmm_modulators_for_config(row, cfg)
        if not bool(mod["trade"]):
            actions.append("hold")
            strategy_names.append("hmm_gate")
            rationales.append("HMM confidence below threshold")
            size_fractions.append(0.0)
            target_profit_pcts.append(0.0)
            max_risk_pcts.append(0.0)
            leg_counts.append(0)
            hmm_confidences.append(mod["confidence"])
            hmm_size_multipliers.append(mod["size_mult"])
            hmm_trade_flags.append(False)
            continue

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
                if "bid_ask_spread" in out.columns and pd.notna(row.get("bid_ask_spread"))
                else None
            ),
            has_major_event=(
                bool(row["has_major_event"])
                if "has_major_event" in out.columns and pd.notna(row.get("has_major_event"))
                else False
            ),
            strike_increment=strike_increment,
            forecast_return=(
                float(row["forecast_return"])
                if "forecast_return" in out.columns and pd.notna(row.get("forecast_return"))
                else (
                    float(row["forecast_return_median"])
                    if "forecast_return_median" in out.columns and pd.notna(row.get("forecast_return_median"))
                    else None
                )
            ),
            realized_vol=(
                float(row["realized_vol"])
                if "realized_vol" in out.columns and pd.notna(row.get("realized_vol"))
                else None
            ),
            use_dynamic_sizing=cfg.use_dynamic_sizing,
            vol_target=cfg.vol_target,
            max_kelly_fraction=cfg.max_kelly_fraction,
            max_capital_fraction=cfg.max_capital_fraction,
        )

        actions.append(decision.action)
        strategy_names.append(decision.strategy_name)
        rationales.append(decision.rationale)
        size_fractions.append(
            (float(decision.size_fraction) * float(mod["size_mult"]))
            if decision.action == "enter" and decision.size_fraction is not None
            else 0.0
        )
        target_profit_pcts.append(float(decision.target_profit_pct or 0.0))
        max_risk_pcts.append(float(decision.max_risk_pct or 0.0))
        leg_counts.append(len(decision.legs))
        hmm_confidences.append(mod["confidence"])
        hmm_size_multipliers.append(mod["size_mult"])
        hmm_trade_flags.append(decision.action == "enter")

    out["roee_action"] = actions
    out["roee_strategy"] = strategy_names
    out["roee_rationale"] = rationales
    out["roee_size_fraction"] = size_fractions
    out["roee_target_profit_pct"] = target_profit_pcts
    out["roee_max_risk_pct"] = max_risk_pcts
    out["roee_leg_count"] = leg_counts
    out["hmm_confidence"] = hmm_confidences
    out["hmm_size_mult"] = hmm_size_multipliers
    out["hmm_trade_allowed"] = hmm_trade_flags

    return out
