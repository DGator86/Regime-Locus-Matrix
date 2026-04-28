from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlm.backtest.fills import FillConfig, exit_fill_price


@dataclass(frozen=True)
class RepricedLeg:
    side: str
    option_type: str
    strike: float
    expiry: str
    bid: float
    ask: float
    mid: float
    mark_value: float
    exit_value: float
    symbol: str | None = None


@dataclass(frozen=True)
class RepriceResult:
    """Outcome of repricing all legs against one chain snapshot."""

    legs: list[RepricedLeg]
    expected_leg_count: int

    @property
    def missing_leg_count(self) -> int:
        return int(self.expected_leg_count - len(self.legs))

    @property
    def is_full(self) -> bool:
        return len(self.legs) == self.expected_leg_count


def _expiry_date_key(expiry_val: object) -> str:
    return str(pd.Timestamp(expiry_val).date())


def _match_leg_snapshot(
    *,
    leg: dict,
    chain_snapshot: pd.DataFrame,
) -> pd.Series | None:
    leg_exp = _expiry_date_key(leg["expiry"])
    subset = chain_snapshot[
        (chain_snapshot["option_type"] == leg["option_type"])
        & (chain_snapshot["strike"] == float(leg["strike"]))
        & (pd.to_datetime(chain_snapshot["expiry"]).dt.date.astype(str) == leg_exp)
    ].copy()

    if subset.empty:
        return None

    subset = subset.sort_values(["spread_pct_mid", "strike"])
    return subset.iloc[0]


def reprice_matched_legs_detailed(
    *,
    matched_legs: list[dict],
    chain_snapshot: pd.DataFrame,
    contract_multiplier: int = 100,
    fill_config: FillConfig | None = None,
) -> RepriceResult:
    """
    Reprices each previously matched leg using current chain snapshot.
    mark_value: mid-based signed valuation
    exit_value: executable exit valuation with slippage
    """
    repriced: list[RepricedLeg] = []
    cfg = fill_config or FillConfig(contract_multiplier=contract_multiplier)

    for leg in matched_legs:
        row = _match_leg_snapshot(leg=leg, chain_snapshot=chain_snapshot)
        if row is None:
            continue

        side = str(leg["side"])
        bid = float(row["bid"])
        ask = float(row["ask"])
        mid = float(row["mid"])

        signed_mid = mid if side == "long" else -mid
        mark_value = signed_mid * contract_multiplier

        executable_exit = exit_fill_price(side=side, bid=bid, ask=ask, config=cfg)
        signed_exit = executable_exit if side == "long" else -executable_exit
        exit_value = signed_exit * contract_multiplier

        repriced.append(
            RepricedLeg(
                side=side,
                option_type=str(leg["option_type"]),
                strike=float(leg["strike"]),
                expiry=str(leg["expiry"]),
                bid=bid,
                ask=ask,
                mid=mid,
                mark_value=mark_value,
                exit_value=exit_value,
                symbol=(
                    str(row["contract_symbol"])
                    if "contract_symbol" in row and pd.notna(row["contract_symbol"])
                    else None
                ),
            )
        )

    return RepriceResult(legs=repriced, expected_leg_count=len(matched_legs))


def reprice_matched_legs(
    *,
    matched_legs: list[dict],
    chain_snapshot: pd.DataFrame,
    contract_multiplier: int = 100,
    fill_config: FillConfig | None = None,
) -> list[RepricedLeg]:
    return reprice_matched_legs_detailed(
        matched_legs=matched_legs,
        chain_snapshot=chain_snapshot,
        contract_multiplier=contract_multiplier,
        fill_config=fill_config,
    ).legs


def aggregate_repriced_mark_value(repriced_legs: list[RepricedLeg]) -> float:
    return float(sum(leg.mark_value for leg in repriced_legs))


def aggregate_repriced_exit_value(repriced_legs: list[RepricedLeg]) -> float:
    return float(sum(leg.exit_value for leg in repriced_legs))


def has_full_reprice(
    matched_legs: list[dict],
    repriced_legs: list[RepricedLeg],
) -> bool:
    return len(matched_legs) == len(repriced_legs)
