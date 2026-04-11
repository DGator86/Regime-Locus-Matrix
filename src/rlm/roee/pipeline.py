from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlm.roee.decision import _finite_float, compute_regime_modulators
from rlm.roee.policy import select_trade
from rlm.roee.regime_safety import attach_regime_safety_columns, build_regime_safety_rationale


@dataclass(frozen=True)
class ROEEConfig:
    hmm_confidence_threshold: float = 0.6
    sizing_multiplier: float = 1.0
    transition_penalty: float = 0.5
    use_dynamic_sizing: bool = False
    vol_target: float = 0.15
    max_kelly_fraction: float = 0.25
    max_capital_fraction: float = 0.5
    regime_adjusted_kelly: bool = True
    high_vol_kelly_multiplier: float = 0.5
    transition_kelly_multiplier: float = 0.75
    calm_trend_kelly_multiplier: float = 1.25
    vault_uncertainty_threshold: float | None = 0.03
    vault_size_multiplier: float = 0.5
    min_regime_train_samples: int = 0
    purge_bars: int = 0
    kronos_confidence_weight: float = 0.4
    hmm_confidence_weight: float = 0.6
    kronos_transition_penalty: float = 0.3


def _hmm_modulators_for_config(row: pd.Series, config: ROEEConfig) -> dict[str, float | bool | str]:
    return compute_regime_modulators(
        row,
        confidence_threshold=config.hmm_confidence_threshold,
        sizing_multiplier=config.sizing_multiplier,
        transition_penalty=config.transition_penalty,
        kronos_confidence_weight=config.kronos_confidence_weight,
        hmm_confidence_weight=config.hmm_confidence_weight,
        kronos_transition_penalty=config.kronos_transition_penalty,
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

    out = attach_regime_safety_columns(
        df,
        min_regime_train_samples=cfg.min_regime_train_samples,
        purge_bars=cfg.purge_bars,
    )

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
    regime_models = []
    regime_confidences = []
    regime_size_multipliers = []
    regime_trade_flags = []
    vault_triggers = []
    vault_size_multipliers = []
    vault_uncertainties = []
    vault_uncertainty_thresholds = []

    for _, row in out.iterrows():
        mod = _hmm_modulators_for_config(row, cfg)
        regime_train_sample_count = int(row.get("regime_train_sample_count", 0) or 0)
        regime_safety_ok = bool(row.get("regime_safety_ok", True))
        if cfg.min_regime_train_samples > 0 and not regime_safety_ok:
            actions.append("hold")
            strategy_names.append("regime_safety_check")
            rationales.append(
                build_regime_safety_rationale(
                    regime_key=str(row.get("regime_key", "")),
                    regime_train_sample_count=regime_train_sample_count,
                    min_regime_train_samples=cfg.min_regime_train_samples,
                    purge_bars=cfg.purge_bars,
                )
            )
            size_fractions.append(0.0)
            target_profit_pcts.append(0.0)
            max_risk_pcts.append(0.0)
            leg_counts.append(0)
            hmm_confidences.append(mod["confidence"])
            hmm_size_multipliers.append(mod["size_mult"])
            hmm_trade_flags.append(False)
            regime_models.append(str(mod["model"]))
            regime_confidences.append(mod["confidence"])
            regime_size_multipliers.append(mod["size_mult"])
            regime_trade_flags.append(False)
            vault_triggers.append(False)
            vault_size_multipliers.append(float(cfg.vault_size_multiplier))
            vault_uncertainties.append(
                float(row["forecast_uncertainty"])
                if "forecast_uncertainty" in out.columns
                and pd.notna(row.get("forecast_uncertainty"))
                else float("nan")
            )
            vault_uncertainty_thresholds.append(
                float(cfg.vault_uncertainty_threshold)
                if cfg.vault_uncertainty_threshold is not None
                else float("nan")
            )
            continue
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
            regime_models.append(str(mod["model"]))
            regime_confidences.append(mod["confidence"])
            regime_size_multipliers.append(mod["size_mult"])
            regime_trade_flags.append(False)
            vault_triggers.append(False)
            vault_size_multipliers.append(float(cfg.vault_size_multiplier))
            vault_uncertainties.append(
                float(row["forecast_uncertainty"])
                if "forecast_uncertainty" in out.columns
                and pd.notna(row.get("forecast_uncertainty"))
                else float("nan")
            )
            vault_uncertainty_thresholds.append(
                float(cfg.vault_uncertainty_threshold)
                if cfg.vault_uncertainty_threshold is not None
                else float("nan")
            )
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
                    if "forecast_return_median" in out.columns
                    and pd.notna(row.get("forecast_return_median"))
                    else None
                )
            ),
            forecast_uncertainty=(
                float(row["forecast_uncertainty"])
                if "forecast_uncertainty" in out.columns
                and pd.notna(row.get("forecast_uncertainty"))
                else None
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
            regime_adjusted_kelly=cfg.regime_adjusted_kelly,
            high_vol_kelly_multiplier=cfg.high_vol_kelly_multiplier,
            transition_kelly_multiplier=cfg.transition_kelly_multiplier,
            calm_trend_kelly_multiplier=cfg.calm_trend_kelly_multiplier,
            vault_uncertainty_threshold=cfg.vault_uncertainty_threshold,
            vault_size_multiplier=cfg.vault_size_multiplier,
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
        regime_models.append(str(mod["model"]))
        regime_confidences.append(mod["confidence"])
        regime_size_multipliers.append(mod["size_mult"])
        regime_trade_flags.append(decision.action == "enter")
        vault_triggers.append(bool(decision.metadata.get("vault_triggered", False)))
        vault_size_multipliers.append(float(decision.metadata.get("vault_size_multiplier", 1.0)))
        vault_uncertainties.append(
            float(decision.metadata["forecast_uncertainty"])
            if "forecast_uncertainty" in decision.metadata
            else float("nan")
        )
        vault_uncertainty_thresholds.append(
            float(decision.metadata["vault_uncertainty_threshold"])
            if "vault_uncertainty_threshold" in decision.metadata
            else float("nan")
        )

    out["roee_action"] = actions
    out["roee_strategy"] = strategy_names
    out["roee_rationale"] = rationales
    out["roee_size_fraction"] = size_fractions
    out["roee_target_profit_pct"] = target_profit_pcts
    out["roee_max_risk_pct"] = max_risk_pcts
    out["roee_leg_count"] = leg_counts
    out["regime_model"] = regime_models
    out["regime_confidence"] = regime_confidences
    out["regime_size_mult"] = regime_size_multipliers
    out["regime_trade_allowed"] = regime_trade_flags
    out["hmm_confidence"] = hmm_confidences
    out["hmm_size_mult"] = hmm_size_multipliers
    out["hmm_trade_allowed"] = hmm_trade_flags
    out["vault_triggered"] = vault_triggers
    out["vault_size_multiplier"] = vault_size_multipliers
    out["vault_forecast_uncertainty"] = vault_uncertainties
    out["vault_uncertainty_threshold"] = vault_uncertainty_thresholds

    return out
