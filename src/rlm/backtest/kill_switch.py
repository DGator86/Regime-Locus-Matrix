"""
Kill switch — hard stop for live and backtest trading.

Two independent triggers halt trading:
    1. Drawdown breach: account equity falls too far from its high-water mark.
    2. Anomaly detection: rolling volatility or loss-run exceeds a threshold
       that signals the market is outside the model's calibrated envelope.

Usage (backtest loop):

    kill_switch = KillSwitch(max_drawdown=-0.15, anomaly_vol_multiplier=3.0)
    for bar in bars:
        kill_switch.update(current_equity=portfolio.equity, realized_vol=bar.vol)
        if kill_switch.halted:
            break  # or: skip trade but continue marking positions

Usage (live):

    if kill_switch.should_halt(current_equity, realized_vol):
        send_alert("Kill switch triggered: " + kill_switch.halt_reason)
        flatten_all_positions()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KillSwitch:
    """
    Stateful kill switch that tracks the high-water mark and anomaly indicators.

    Parameters
    ----------
    max_drawdown:
        Maximum tolerated drawdown from the high-water mark (negative fraction,
        e.g. -0.15 for 15%). Trading halts when this is breached.
    anomaly_vol_multiplier:
        Halt when the current realized_vol exceeds
        baseline_vol * anomaly_vol_multiplier.  Set to None to disable.
    consecutive_loss_limit:
        Halt after this many consecutive losing trades. Set to None to disable.
    """

    max_drawdown: float = -0.15
    anomaly_vol_multiplier: float | None = 3.0
    consecutive_loss_limit: int | None = 5

    # Internal state — not constructor args.
    _high_water_mark: float = field(default=0.0, init=False, repr=False)
    _baseline_vol: float | None = field(default=None, init=False, repr=False)
    _consecutive_losses: int = field(default=0, init=False, repr=False)
    _halted: bool = field(default=False, init=False, repr=False)
    _halt_reason: str = field(default="", init=False, repr=False)

    def reset(self, initial_equity: float) -> None:
        """Initialise or reset the high-water mark. Call before the first bar."""
        self._high_water_mark = float(initial_equity)
        self._baseline_vol = None
        self._consecutive_losses = 0
        self._halted = False
        self._halt_reason = ""

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def update(
        self,
        current_equity: float,
        realized_vol: float | None = None,
        last_trade_pnl: float | None = None,
    ) -> bool:
        """
        Update state and return True if the kill switch just fired.

        Parameters
        ----------
        current_equity:
            Current portfolio NAV / equity value.
        realized_vol:
            Current bar realized volatility (annualised). Used for anomaly detection.
        last_trade_pnl:
            PnL of the most recently closed trade. Used for consecutive-loss tracking.

        Returns True on the bar the halt is triggered; False otherwise.
        Already-halted instances return True immediately without re-evaluating.
        """
        if self._halted:
            return True

        equity = float(current_equity)

        if equity > self._high_water_mark:
            self._high_water_mark = equity
            # Reset consecutive-loss counter on new equity high.
            self._consecutive_losses = 0

        # --- Drawdown gate ---
        if self._high_water_mark > 0:
            drawdown = (equity - self._high_water_mark) / self._high_water_mark
            if drawdown < self.max_drawdown:
                return self._trigger(f"Drawdown {drawdown:.2%} breached limit {self.max_drawdown:.2%}.")

        # --- Anomaly: realized vol spike ---
        if realized_vol is not None and np.isfinite(realized_vol) and self.anomaly_vol_multiplier is not None:
            vol = float(realized_vol)
            if self._baseline_vol is None:
                self._baseline_vol = vol
            else:
                self._baseline_vol = 0.95 * self._baseline_vol + 0.05 * vol
            if self._baseline_vol > 0 and vol > self._baseline_vol * self.anomaly_vol_multiplier:
                return self._trigger(
                    f"Vol anomaly: current {vol:.4f} > "
                    f"{self.anomaly_vol_multiplier}× baseline {self._baseline_vol:.4f}."
                )

        # --- Consecutive loss gate ---
        if last_trade_pnl is not None and self.consecutive_loss_limit is not None:
            if float(last_trade_pnl) < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0
            if self._consecutive_losses >= self.consecutive_loss_limit:
                return self._trigger(
                    f"{self._consecutive_losses} consecutive losing trades " f"(limit {self.consecutive_loss_limit})."
                )

        return False

    def should_halt(
        self,
        current_equity: float,
        realized_vol: float | None = None,
        last_trade_pnl: float | None = None,
    ) -> bool:
        """Stateless check — returns True if the kill switch would fire given the inputs."""
        if self._halted:
            return True
        self.update(current_equity, realized_vol, last_trade_pnl)
        return self._halted

    def _trigger(self, reason: str) -> bool:
        self._halted = True
        self._halt_reason = reason
        return True
