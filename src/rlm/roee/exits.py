from __future__ import annotations


def should_exit_for_profit(
    pnl_pct: float,
    target_profit_pct: float,
) -> bool:
    return pnl_pct >= target_profit_pct


def should_exit_for_zone_breach(
    realized_price: float,
    lower_1s: float,
    upper_1s: float,
) -> bool:
    return (realized_price < lower_1s) or (realized_price > upper_1s)


def should_exit_for_regime_flip(
    entry_regime_key: str,
    current_regime_key: str,
) -> bool:
    # Regime key format: "direction|volatility|liquidity|dealer_flow".
    # Only the direction component (index 0) warrants an immediate exit —
    # vol/liquidity noise would otherwise flush positions every 1-2 bars.
    #
    # Two hold rules prevent premature exits:
    #
    # 1. "transition" (S_D ±0.3–0.6) is signal uncertainty, not a reversal.
    #    A bull position whose S_D dips to 0.55 should not be flushed — it
    #    would exit at entry-spread cost before any P&L can develop.
    #
    # 2. Directional positions (bull/bear) hold through a range stall.
    #    If S_D falls from 0.65 → 0.25 the thesis has weakened, but the
    #    -50% hard stop and 45% profit target already manage that outcome.
    #    Exiting on "bull→range" cuts potential winners and duplicates the
    #    stop-loss function.  Non-directional range positions (iron condors /
    #    strangles) still exit immediately when a true breakout occurs
    #    (range→bull or range→bear) because the short leg is then at risk.
    if "|" in entry_regime_key and "|" in current_regime_key:
        entry_dir = entry_regime_key.split("|")[0]
        current_dir = current_regime_key.split("|")[0]
        if entry_dir == "transition" or current_dir == "transition":
            return False
        if entry_dir in {"bull", "bear"} and current_dir == "range":
            return False
        return entry_dir != current_dir
    return entry_regime_key != current_regime_key


def should_exit_for_stop_loss(
    pnl_pct: float,
    stop_loss_pct: float,
) -> bool:
    """Exit if loss exceeds the stop loss threshold (e.g., -0.50)."""
    return pnl_pct <= stop_loss_pct


def should_exit_for_time_stop(
    dte_remaining: float,
    min_dte_threshold: float,
) -> bool:
    """Exit if the time remaining until expiry is below the threshold."""
    return dte_remaining <= min_dte_threshold
