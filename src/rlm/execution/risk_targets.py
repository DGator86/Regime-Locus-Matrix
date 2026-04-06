"""Dollar-based exit levels for **long-premium / debit-style** spreads using mid marks.

Convention (matches :func:`~rlm.roee.chain_match.estimate_mark_value_from_matched_legs`):

- ``V`` = net mid liquidation value of the combo (× contract multiplier already applied).
- ``D`` = positive debit paid to open (from ask/bid entry cost).
- We snapshot ``V0`` at decision time and compare live ``V`` to thresholds anchored at ``V0``.

This is a **monitoring heuristic**, not a guarantee of fill prices (uses mids; closes use bid/ask).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpreadExitThresholds:
    """Absolute mid-mark levels for the same signed combo valuation as ``V0``."""

    v_take_profit: float
    v_hard_stop: float
    v_trail_activate: float
    trail_retrace_frac: float


def build_spread_exit_thresholds(
    *,
    v0: float,
    entry_debit: float,
    target_profit_pct: float,
    stop_loss_frac_of_debit: float = 0.5,
    trail_activate_frac_of_debit: float = 0.15,
    trail_retrace_frac_from_peak: float = 0.25,
) -> SpreadExitThresholds:
    """
    ``target_profit_pct`` follows :class:`~rlm.types.options.TradeCandidate` (e.g. 0.50 → +50% of ``D`` vs ``V0``).

    ``stop_loss_frac_of_debit`` default 0.5 → exit if mid mark falls by ``0.5 * D`` from ``V0``.
    """
    d = float(entry_debit)
    if d < 0:
        d = abs(d)
    tp = float(target_profit_pct)
    return SpreadExitThresholds(
        v_take_profit=float(v0) + tp * d,
        v_hard_stop=float(v0) - float(stop_loss_frac_of_debit) * d,
        v_trail_activate=float(v0) + float(trail_activate_frac_of_debit) * d,
        trail_retrace_frac=float(trail_retrace_frac_from_peak),
    )


def trailing_stop_from_peak(peak_v: float, retrace_frac: float) -> float:
    return float(peak_v) * (1.0 - float(retrace_frac))
