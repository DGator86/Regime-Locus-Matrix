from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rlm.data.option_chain import select_nearest_expiry_slice
from rlm.types.options import TradeDecision


@dataclass(frozen=True)
class MatchedLeg:
    side: str
    option_type: str
    strike: float
    expiry: str
    bid: float
    ask: float
    mid: float
    symbol: str | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    iv: float | None = None


def _row_to_matched_leg(row: pd.Series, side: str) -> MatchedLeg:
    return MatchedLeg(
        side=side,
        option_type=str(row["option_type"]),
        strike=float(row["strike"]),
        expiry=str(pd.Timestamp(row["expiry"]).date()),
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        mid=float(row["mid"]),
        symbol=(str(row["contract_symbol"]) if "contract_symbol" in row and pd.notna(row["contract_symbol"]) else None),
        delta=float(row["delta"]) if "delta" in row and pd.notna(row["delta"]) else None,
        gamma=float(row["gamma"]) if "gamma" in row and pd.notna(row["gamma"]) else None,
        theta=float(row["theta"]) if "theta" in row and pd.notna(row["theta"]) else None,
        vega=float(row["vega"]) if "vega" in row and pd.notna(row["vega"]) else None,
        iv=float(row["iv"]) if "iv" in row and pd.notna(row["iv"]) else None,
    )


def find_nearest_contract(
    chain_slice: pd.DataFrame,
    option_type: str,
    target_strike: float,
) -> pd.Series | None:
    subset = chain_slice[chain_slice["option_type"] == option_type].copy()
    if subset.empty:
        return None

    subset["strike_distance"] = (subset["strike"] - target_strike).abs()
    if "mid" not in subset.columns:
        subset["mid"] = (subset["bid"].astype(float) + subset["ask"].astype(float)) / 2.0
    if "spread_pct_mid" not in subset.columns:
        spread = subset["ask"].astype(float) - subset["bid"].astype(float)
        subset["spread_pct_mid"] = np.where(subset["mid"] > 0, spread / subset["mid"], np.nan)
    subset = subset.sort_values(["strike_distance", "spread_pct_mid", "strike"])
    return subset.iloc[0]


def _leg_expiry_selector(expiry: str | None) -> str:
    return str(expiry or "").strip().lower()


def decision_needs_multi_expiry_chain(decision: TradeDecision) -> bool:
    return any(_leg_expiry_selector(leg.expiry) in {"near", "front", "far", "back"} for leg in decision.legs)


def select_chain_slice_for_decision(
    chain: pd.DataFrame,
    decision: TradeDecision,
    *,
    dte_min: int | None = None,
    dte_max: int | None = None,
) -> pd.DataFrame:
    candidate = decision.candidate
    if candidate is None:
        return chain.copy()

    dte_min = int(candidate.target_dte_min if dte_min is None else dte_min)
    dte_max = int(candidate.target_dte_max if dte_max is None else dte_max)
    if not decision_needs_multi_expiry_chain(decision):
        return select_nearest_expiry_slice(chain, dte_min=dte_min, dte_max=dte_max)

    if "dte" not in chain.columns:
        return chain.copy()
    eligible = chain[(chain["dte"] >= dte_min) & (chain["dte"] <= dte_max)].copy()
    return eligible


def _filter_chain_for_leg_expiry(chain_slice: pd.DataFrame, expiry: str | None) -> pd.DataFrame:
    selector = _leg_expiry_selector(expiry)
    if not selector:
        return chain_slice

    if "expiry" not in chain_slice.columns or chain_slice.empty:
        return chain_slice.iloc[0:0].copy()

    expiry_ts = pd.to_datetime(chain_slice["expiry"], errors="coerce")
    if selector in {"near", "front"}:
        valid = expiry_ts.dropna()
        if valid.empty:
            return chain_slice.iloc[0:0].copy()
        target = valid.min().normalize()
        return chain_slice[expiry_ts.dt.normalize() == target].copy()
    if selector in {"far", "back"}:
        valid = expiry_ts.dropna()
        if valid.empty:
            return chain_slice.iloc[0:0].copy()
        target = valid.max().normalize()
        return chain_slice[expiry_ts.dt.normalize() == target].copy()

    try:
        target = pd.Timestamp(selector).normalize()
    except (TypeError, ValueError):
        return chain_slice.iloc[0:0].copy()
    return chain_slice[expiry_ts.dt.normalize() == target].copy()


def match_legs_to_chain(
    *,
    decision: TradeDecision,
    chain_slice: pd.DataFrame,
) -> TradeDecision:
    """
    Replaces abstract legs with actual matched chain contracts.
    """
    if decision.action != "enter" or not decision.legs:
        return decision

    matched_legs: list[MatchedLeg] = []

    for leg in decision.legs:
        leg_chain = _filter_chain_for_leg_expiry(chain_slice, leg.expiry)
        row = find_nearest_contract(
            chain_slice=leg_chain,
            option_type=leg.option_type,
            target_strike=leg.strike,
        )
        if row is None:
            expiry_hint = f" expiry={leg.expiry}" if leg.expiry else ""
            return TradeDecision(
                action="skip",
                strategy_name=decision.strategy_name,
                regime_key=decision.regime_key,
                rationale=f"Chain match failed for {leg.option_type}{expiry_hint} strike≈{leg.strike}.",
                candidate=decision.candidate,
                metadata=decision.metadata,
            )

        matched_legs.append(_row_to_matched_leg(row, side=leg.side))

    # Reject combos where two legs resolve to the exact same contract (same option_type,
    # strike, and expiry).  Such a position has zero net exposure and IBKR rejects it
    # as a "riskless combination" (error 10087).
    seen_contracts: set[tuple[str, float, str]] = set()
    for ml in matched_legs:
        key = (ml.option_type, ml.strike, ml.expiry)
        if key in seen_contracts:
            return TradeDecision(
                action="skip",
                strategy_name=decision.strategy_name,
                regime_key=decision.regime_key,
                rationale=f"Duplicate contract for {key} — riskless combination rejected.",
                candidate=decision.candidate,
                metadata=decision.metadata,
            )
        seen_contracts.add(key)

    new_metadata = dict(decision.metadata)
    new_metadata["matched_legs"] = [m.__dict__ for m in matched_legs]

    return TradeDecision(
        action=decision.action,
        strategy_name=decision.strategy_name,
        regime_key=decision.regime_key,
        rationale=decision.rationale,
        size_fraction=decision.size_fraction,
        target_profit_pct=decision.target_profit_pct,
        max_risk_pct=decision.max_risk_pct,
        candidate=decision.candidate,
        legs=decision.legs,
        metadata=new_metadata,
    )


def estimate_entry_cost_from_matched_legs(
    decision: TradeDecision,
    contract_multiplier: int = 100,
) -> float:
    """
    Positive cost means capital outflow.
    Long legs pay ask.
    Short legs receive bid.
    Net debit > 0, net credit < 0 in this convention.
    """
    matched = decision.metadata.get("matched_legs", [])
    if not matched:
        return np.nan

    total = 0.0
    for leg in matched:
        if leg["side"] == "long":
            total += float(leg["ask"]) * contract_multiplier
        elif leg["side"] == "short":
            total -= float(leg["bid"]) * contract_multiplier
        else:
            raise ValueError(f"Unknown leg side: {leg['side']}")

    return total


def estimate_mark_value_from_matched_legs(
    matched_legs: list[dict],
    contract_multiplier: int = 100,
) -> float:
    """
    Mid-mark valuation for open positions.
    Long legs valued at +mid.
    Short legs valued at -mid.
    """
    total = 0.0
    for leg in matched_legs:
        signed_mid = float(leg["mid"]) if leg["side"] == "long" else -float(leg["mid"])
        total += signed_mid * contract_multiplier
    return total
