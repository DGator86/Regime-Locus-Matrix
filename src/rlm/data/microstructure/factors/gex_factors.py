"""
GEX-based FactorCalculator for the RLM microstructure layer.

Produces the following factor columns that slot into the existing
``FactorPipeline`` / ``S_D`` / ``S_V`` / ``S_L`` / ``S_G`` composite scores:

  ``gex_net_total``      — Total net dealer gamma exposure (absolute $, dealer POV)
  ``gex_sign``           — +1 if dealers are long gamma, −1 if short gamma
  ``gex_normalized``     — Net GEX normalised by 60-bar rolling absolute maximum
  ``gex_flip_distance``  — (spot − flip_strike) / spot  (signed)
  ``gex_call_put_ratio`` — call_gex / |put_gex|  (>1 → calls dominate)

Wiring into FactorPipeline
--------------------------
Add a ``GEXFactors`` instance to ``FactorPipeline.specs()``::

    from rlm.data.microstructure.factors.gex_factors import GEXFactors
    from rlm.features.factors.pipeline import FactorPipeline

    pipeline = FactorPipeline(extra_calculators=[GEXFactors(db, symbol="SPY")])
    result = pipeline.run(bars_df)

The bars_df index must be a DatetimeIndex and the GEX lake must contain data
for the same dates.  Missing dates are forward-filled (GEX changes slowly
intraday).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind

if TYPE_CHECKING:
    from rlm.data.microstructure.database.query import MicrostructureDB

logger = logging.getLogger(__name__)

_GEX_NORM_WINDOW = 60  # rolling window for GEX normalisation (trading days)


class GEXFactors(FactorCalculator):
    """
    Dealer Gamma Exposure factors from the microstructure lake.

    Parameters
    ----------
    db      : MicrostructureDB instance (provides DuckDB access)
    symbol  : Underlying ticker (e.g. "SPY")
    """

    def __init__(self, db: "MicrostructureDB", symbol: str) -> None:
        self._db = db
        self._symbol = symbol.upper()

    def specs(self) -> list[FactorSpec]:
        return [
            # gex_sign: +1 long gamma / −1 short gamma → contributes to DEALER_FLOW
            FactorSpec(
                name="gex_sign",
                category=FactorCategory.DEALER_FLOW,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=2.0,
            ),
            # gex_normalized: signed, scaled to [−1, +1] via rolling max
            FactorSpec(
                name="gex_normalized",
                category=FactorCategory.DEALER_FLOW,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.5,
                k=1.5,
            ),
            # gex_flip_distance: signed distance to GEX flip level
            # Positive → spot above flip (long-gamma territory)
            # Negative → spot below flip (short-gamma territory)
            FactorSpec(
                name="gex_flip_distance",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.03,  # 3% typical distance
                k=1.0,
            ),
            # gex_call_put_ratio: calls / puts GEX imbalance → liquidity signal
            FactorSpec(
                name="gex_call_put_ratio",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.0,
            ),
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach GEX factor columns to *df*.

        *df* must have a DatetimeIndex.  GEX data is loaded from the lake for
        the date range covered by *df*, then reindexed + forward-filled onto
        the bar timestamps.
        """
        if df.empty:
            return _empty_gex_columns(df)

        start = df.index.min().strftime("%Y-%m-%d")
        end = df.index.max().strftime("%Y-%m-%d")

        try:
            gex_raw = self._db.load_gex_surface(self._symbol, start, end, net_gex_only=False)
        except Exception as exc:
            logger.warning("GEXFactors: failed to load GEX data (%s). Using NaN.", exc)
            return _empty_gex_columns(df)

        if gex_raw.empty:
            logger.warning("GEXFactors: no GEX data for %s %s → %s", self._symbol, start, end)
            return _empty_gex_columns(df)

        # Aggregate per timestamp (sum across all strikes/expiries)
        gex_ts = (
            gex_raw.groupby("timestamp")
            .agg(
                net_gex=("net_gex", "sum"),
                call_gex=("call_gex", "sum"),
                put_gex=("put_gex", "sum"),
                underlying_price=("underlying_price", "first"),
                gex=("gex", "sum"),
            )
            .reset_index()
            .sort_values("timestamp")
            .set_index("timestamp")
        )
        gex_ts.index = pd.DatetimeIndex(gex_ts.index)

        # Load GEX flip strikes per timestamp
        try:
            gex_detail = self._db.load_gex_surface(self._symbol, start, end, net_gex_only=False)
            flip_per_ts = _compute_flip_per_timestamp(gex_detail)
        except Exception:
            flip_per_ts = pd.Series(dtype=float)

        # Build factor series, aligned to df's daily index (date-level join)
        out = df.copy()

        gex_ts["_date"] = pd.to_datetime(gex_ts.index).date
        date_agg = gex_ts.groupby("_date").agg(
            net_gex=("net_gex", "last"),
            call_gex=("call_gex", "last"),
            put_gex=("put_gex", "last"),
            underlying_price=("underlying_price", "last"),
        )
        date_agg.index = pd.to_datetime(date_agg.index)

        # Reindex onto df's date index, fill forward
        df_dates = pd.to_datetime(df.index.date)
        net_gex = date_agg["net_gex"].reindex(df_dates).ffill().values
        call_gex = date_agg["call_gex"].reindex(df_dates).ffill().values
        put_gex = date_agg["put_gex"].reindex(df_dates).ffill().values
        spot_gex = date_agg["underlying_price"].reindex(df_dates).ffill().values

        # Flip distance (reindex flip series similarly)
        if not flip_per_ts.empty:
            flip_per_ts.index = pd.to_datetime(flip_per_ts.index.date)
            flip_strikes = flip_per_ts.reindex(df_dates).ffill().values
        else:
            flip_strikes = np.full(len(df), float("nan"))

        # Factor columns
        out["gex_net_total"] = net_gex
        out["gex_sign"] = np.sign(net_gex).astype(float)

        # Normalised GEX: roll max abs over _GEX_NORM_WINDOW
        abs_gex = pd.Series(np.abs(net_gex))
        roll_max = abs_gex.rolling(_GEX_NORM_WINDOW, min_periods=5).max().fillna(abs_gex.abs())
        out["gex_normalized"] = np.where(roll_max > 0, net_gex / roll_max.values, 0.0)

        # Flip distance
        out["gex_flip_distance"] = np.where(
            np.isfinite(flip_strikes) & (spot_gex > 0),
            (spot_gex - flip_strikes) / spot_gex,
            float("nan"),
        )

        # Call / put ratio
        abs_put = np.abs(put_gex)
        out["gex_call_put_ratio"] = np.where(abs_put > 0, call_gex / abs_put, float("nan"))

        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_gex_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (
        "gex_net_total",
        "gex_sign",
        "gex_normalized",
        "gex_flip_distance",
        "gex_call_put_ratio",
    ):
        out[col] = float("nan")
    return out


def _compute_flip_per_timestamp(gex_df: pd.DataFrame) -> pd.Series:
    """Return a Series of flip strikes indexed by timestamp."""
    from rlm.data.microstructure.calculators.gex import gex_flip_level

    flips: dict[Any, float] = {}
    for ts, grp in gex_df.groupby("timestamp"):
        flip = gex_flip_level(grp)
        flips[ts] = flip if flip is not None else float("nan")
    return pd.Series(flips)
