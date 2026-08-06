"""Dollar-based exit levels for debit- and credit-style spreads using mid marks.

Convention (matches :func:`~rlm.roee.chain_match.estimate_mark_value_from_matched_legs`):

- ``V`` = net mid liquidation value of the combo (× contract multiplier already applied).
  Debits are typically ``V > 0``; credits are typically ``V < 0`` (short legs dominate).
- ``D`` = absolute entry debit/credit magnitude (pipeline passes ``abs(entry_debit)``).
- We snapshot ``V0`` at decision time and compare live ``V`` to thresholds anchored at ``V0``.
  Favourable marks move **up** the number line for both styles (debit: higher premium;
  credit: less-negative close cost).

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
    min_trail_exit_v: float


def build_spread_exit_thresholds(
    *,
    v0: float,
    entry_debit: float,
    target_profit_pct: float,
    stop_loss_frac_of_debit: float = 0.5,
    trail_activate_frac_of_debit: float = 0.30,
    trail_retrace_frac_from_peak: float = 0.20,
    min_trail_profit_frac_of_debit: float = 0.08,
) -> SpreadExitThresholds:
    """
    ``target_profit_pct`` follows :class:`~rlm.types.options.TradeCandidate` (e.g. 0.50 → +50% of ``D`` vs ``V0``).

    ``stop_loss_frac_of_debit`` default 0.5 → exit if mid mark falls by ``0.5 * D`` from ``V0``.
    """
    d = float(entry_debit)
    if d < 0:
        d = abs(d)
    tp = float(target_profit_pct)
    min_trail = float(v0) + float(min_trail_profit_frac_of_debit) * d
    return SpreadExitThresholds(
        v_take_profit=float(v0) + tp * d,
        v_hard_stop=float(v0) - float(stop_loss_frac_of_debit) * d,
        v_trail_activate=float(v0) + float(trail_activate_frac_of_debit) * d,
        trail_retrace_frac=float(trail_retrace_frac_from_peak),
        min_trail_exit_v=min_trail,
    )


def trailing_stop_from_peak(peak_v: float, retrace_frac: float) -> float:
    """Trail stop placed an adverse retrace *below* the favourable peak mark.

    Uses ``peak - abs(peak) * retrace`` so debit peaks (positive) stop below the
    peak and credit peaks (negative) stop *more negative* than the peak. The
    previous ``peak * (1 - retrace)`` formula put credit stops *above* the peak,
    so ``should_trailing_stop_exit`` fired on the same tick the trail armed.
    """
    peak = float(peak_v)
    return peak - abs(peak) * float(retrace_frac)


def should_trailing_stop_exit(
    *,
    v: float,
    peak_v: float,
    retrace_frac: float,
    min_exit_v: float,
) -> bool:
    """True only when mark fell through the trail **and** is still at/above the profit floor."""
    tstop = trailing_stop_from_peak(peak_v, retrace_frac)
    return float(v) < tstop and float(v) >= float(min_exit_v)
