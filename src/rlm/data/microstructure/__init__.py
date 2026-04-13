"""
RLM Microstructure Layer
========================

High-frequency Greeks, Gamma Exposure (GEX), and Implied Volatility (IV) surface
infrastructure for the Regime-Locus-Matrix.

Architecture
------------
  data/microstructure/
  ├── underlying/{symbol}/1s/      — 5-second OHLCV bars
  ├── options/{symbol}/
  │   ├── greeks_snapshots/        — Per-contract Greeks every 5 seconds
  │   └── derived/
  │       ├── gex_surface/         — Dealer Gamma Exposure surface
  │       └── iv_surface/          — Interpolated IV surface
  └── metadata/

Sub-packages
------------
  collectors   : IBKR real-time data collectors (underlying + options)
  calculators  : Greek computation, GEX aggregation, IV interpolation
  database     : DuckDB query interface (MicrostructureDB)
  factors      : FactorCalculator plug-ins for FactorPipeline (GEX, IV)

Quick start::

    from rlm.data.microstructure.database.query import MicrostructureDB
    from rlm.data.microstructure.factors import GEXFactors, IVSurfaceFactors

    db = MicrostructureDB()
    ctx = db.microstructure_regime_context("SPY", "2025-06-10 15:30:00")
    # → {'gex_net_total': ..., 'gex_flip_strike': ..., 'iv_atm': ..., ...}
"""

from rlm.data.microstructure.database.query import MicrostructureDB
from rlm.data.microstructure.calculators.greeks import GreekBundle, full_greeks_row, solve_iv
from rlm.data.microstructure.calculators.gex import (
    build_gex_surface_from_df,
    gex_flip_level,
    aggregate_gex_profile,
)
from rlm.data.microstructure.calculators.iv_surface import build_iv_surface, query_iv_surface
from rlm.data.microstructure.factors.gex_factors import GEXFactors
from rlm.data.microstructure.factors.iv_surface_factors import IVSurfaceFactors

__all__ = [
    # Database
    "MicrostructureDB",
    # Greeks
    "GreekBundle",
    "full_greeks_row",
    "solve_iv",
    # GEX
    "build_gex_surface_from_df",
    "gex_flip_level",
    "aggregate_gex_profile",
    # IV Surface
    "build_iv_surface",
    "query_iv_surface",
    # Factors
    "GEXFactors",
    "IVSurfaceFactors",
]
