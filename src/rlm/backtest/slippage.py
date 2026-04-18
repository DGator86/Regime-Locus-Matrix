from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SlippageConfig:
    per_contract_flat: float = 0.0
    spread_fraction: float = 0.25
    min_slippage: float = 0.0
    # Volatility scaling: slippage *= (1 + vol_sensitivity * realized_vol)
    # Set to 0 to disable. A value of 3.0 triples slippage at 100% annualised vol.
    vol_sensitivity: float = 3.0
    # Probability of a partial fill (0 = always full fill, 1 = never fills).
    partial_fill_probability: float = 0.0
    # Number of bars to delay fill execution (latency simulation).
    latency_bars: int = 0


def compute_leg_slippage(
    *,
    bid: float,
    ask: float,
    config: SlippageConfig | None = None,
    realized_vol: float | None = None,
) -> float:
    """
    Returns slippage in option-price units, per contract, per leg.

    When ``realized_vol`` is provided and ``config.vol_sensitivity > 0``,
    slippage scales with volatility:
        slippage *= (1 + vol_sensitivity * realized_vol)

    This prevents backtest fantasy returns during high-vol regimes where
    real spreads widen and fills deteriorate significantly.
    """
    cfg = config or SlippageConfig()
    spread = max(ask - bid, 0.0)
    base_slip = max(cfg.min_slippage, spread * cfg.spread_fraction)
    slip = base_slip + cfg.per_contract_flat

    if realized_vol is not None and realized_vol > 0.0 and cfg.vol_sensitivity > 0.0:
        vol_multiplier = 1.0 + cfg.vol_sensitivity * float(realized_vol)
        slip *= vol_multiplier

    return slip
