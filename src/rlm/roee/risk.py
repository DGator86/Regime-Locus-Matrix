from __future__ import annotations


def should_require_defined_risk(
    liquidity_score: float,
    dealer_flow_score: float,
) -> bool:
    return (liquidity_score < 0.0) or (dealer_flow_score < -0.4)


def is_tradeable_environment(
    *,
    has_major_event: bool = False,
    bid_ask_spread_pct: float | None = None,
    volume_ratio: float | None = None,
    regime_transition: bool = False,
    max_spread_pct: float = 0.05,
    min_volume_ratio: float = 0.5,
) -> tuple[bool, str]:
    """
    Unified no-trade-zone gate.

    Returns (tradeable, reason). A single False from any condition blocks the trade.
    Checks (in order):
      1. Major event window
      2. Spread quality
      3. Low volume
      4. Regime transition
    """
    if has_major_event:
        return False, "Major event risk filter active."

    if bid_ask_spread_pct is not None and bid_ask_spread_pct > max_spread_pct:
        return False, f"Spread {bid_ask_spread_pct:.4f} exceeds max {max_spread_pct:.4f}."

    if volume_ratio is not None and volume_ratio < min_volume_ratio:
        return False, f"Volume ratio {volume_ratio:.2f} below minimum {min_volume_ratio:.2f}."

    if regime_transition:
        return False, "Regime transition in progress — no trade zone."

    return True, ""


# ---------------------------------------------------------------------------
# Legacy helpers kept for callers that haven't migrated yet.
# ---------------------------------------------------------------------------

def should_skip_for_event_risk(has_major_event: bool) -> bool:
    return bool(has_major_event)


def spread_quality_ok(
    bid_ask_spread_pct: float | None,
    max_spread_pct: float = 0.05,
) -> bool:
    if bid_ask_spread_pct is None:
        return True
    return bid_ask_spread_pct <= max_spread_pct
