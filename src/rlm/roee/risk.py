from __future__ import annotations


def should_require_defined_risk(
    liquidity_score: float,
    dealer_flow_score: float,
) -> bool:
    return (liquidity_score < 0.0) or (dealer_flow_score < -0.4)


def should_skip_for_event_risk(
    has_major_event: bool,
) -> bool:
    return bool(has_major_event)


def spread_quality_ok(
    bid_ask_spread_pct: float | None,
    max_spread_pct: float = 0.05,
) -> bool:
    if bid_ask_spread_pct is None:
        return True
    return bid_ask_spread_pct <= max_spread_pct
