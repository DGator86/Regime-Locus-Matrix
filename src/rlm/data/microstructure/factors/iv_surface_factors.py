"""
IV Surface FactorCalculator for the RLM microstructure layer.

Derives regime-sensitive factors from the interpolated implied volatility surface:

  ``iv_atm_30d``         — ATM IV at 30-day DTE (volatility level)
  ``iv_skew_25d``        — Put-call IV skew at 25-delta (put IV - call IV)
  ``iv_term_ratio``      — Short-term / long-term IV ratio (term structure slope)
  ``iv_surface_change``  — Day-over-day change in ATM 30-day IV
  ``iv_vol_of_vol``      — Rolling std-dev of ATM 30-day IV (vol-of-vol proxy)

These supplement the existing ``vix_ratio`` / ``vvix_ratio`` factors in
``rlm.factors.volatility`` with a symbol-specific surface read.

Wiring::

    from rlm.data.microstructure.factors.iv_surface_factors import IVSurfaceFactors
    pipeline = FactorPipeline(extra_calculators=[IVSurfaceFactors(db, symbol="SPY")])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind

if TYPE_CHECKING:
    from rlm.data.microstructure.database.query import MicrostructureDB

logger = logging.getLogger(__name__)

_IV_VOL_WINDOW = 20  # rolling window for vol-of-vol calculation


class IVSurfaceFactors(FactorCalculator):
    """
    Implied volatility surface factors from the microstructure lake.

    Parameters
    ----------
    db              : MicrostructureDB instance
    symbol          : Underlying ticker (e.g. "SPY")
    atm_dte         : DTE used for ATM IV reference (default 30)
    skew_delta_otm  : OTM delta for skew calculation (default 0.10 → 10-delta)
    term_short_dte  : Short DTE for term structure ratio (default 7)
    term_long_dte   : Long DTE for term structure ratio (default 90)
    """

    def __init__(
        self,
        db: "MicrostructureDB",
        symbol: str,
        *,
        atm_dte: float = 30.0,
        skew_delta_otm: float = 0.10,
        term_short_dte: float = 7.0,
        term_long_dte: float = 90.0,
    ) -> None:
        self._db = db
        self._symbol = symbol.upper()
        self._atm_dte = atm_dte
        self._skew_otm = skew_delta_otm
        self._short_dte = term_short_dte
        self._long_dte = term_long_dte

    def specs(self) -> list[FactorSpec]:
        return [
            # ATM vol level: VOLATILITY category
            FactorSpec(
                name="iv_atm_30d",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.20,  # 20% IV is neutral baseline
                k=1.5,
                invert=True,  # Higher IV → lower score (bearish environment)
            ),
            # Put-call skew: VOLATILITY (skew surge → regime shift)
            FactorSpec(
                name="iv_skew_25d",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.05,  # 5 vol-point skew is "significant"
                k=1.0,
                invert=True,  # Higher skew → downside fear → bearish
            ),
            # Term structure: VOLATILITY (inverted term = stress)
            FactorSpec(
                name="iv_term_ratio",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,  # Flat term structure
                k=2.0,
                invert=True,  # ratio > 1 → front elevated → bearish
            ),
            # Day-over-day IV change: DIRECTION (IV rising → bearish pressure)
            FactorSpec(
                name="iv_surface_change",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.02,  # 2 vol-point move is 1 standard-scale unit
                k=1.0,
                invert=True,  # Rising IV → negative direction signal
            ),
            # Vol-of-vol: VOLATILITY regime (elevated = unstable)
            FactorSpec(
                name="iv_vol_of_vol",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.03,  # 3% vol-of-vol baseline
                k=1.5,
                invert=True,
            ),
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute IV surface factor columns aligned to *df*'s DatetimeIndex.
        """
        if df.empty:
            return _empty_iv_columns(df)

        start = df.index.min().strftime("%Y-%m-%d")
        end = df.index.max().strftime("%Y-%m-%d")

        try:
            iv_surfaces = self._db.load_iv_surface_range(self._symbol, start, end)
        except Exception as exc:
            logger.warning("IVSurfaceFactors: failed to load IV surfaces (%s). Using NaN.", exc)
            return _empty_iv_columns(df)

        if iv_surfaces.empty:
            logger.warning(
                "IVSurfaceFactors: no IV surfaces for %s %s→%s", self._symbol, start, end
            )
            return _empty_iv_columns(df)

        # Build per-date IV metrics from the stored surface grid
        from rlm.data.microstructure.calculators.iv_surface import query_iv_surface, skew_at_dte

        daily_metrics: list[dict[str, Any]] = []
        for ts, surface in iv_surfaces.groupby("timestamp"):
            iv_atm = query_iv_surface(surface, moneyness=1.0, dte=self._atm_dte)
            iv_skew = skew_at_dte(surface, dte=self._atm_dte, delta_otm=self._skew_otm)
            iv_short = query_iv_surface(surface, moneyness=1.0, dte=self._short_dte)
            iv_long = query_iv_surface(surface, moneyness=1.0, dte=self._long_dte)
            term_ratio = (iv_short / iv_long) if (iv_long and iv_long > 0) else float("nan")
            daily_metrics.append(
                {
                    "date": pd.Timestamp(ts).date(),
                    "iv_atm_30d": iv_atm,
                    "iv_skew_25d": iv_skew,
                    "iv_term_ratio": term_ratio,
                }
            )

        metrics_df = pd.DataFrame(daily_metrics).set_index("date")
        metrics_df.index = pd.DatetimeIndex(metrics_df.index)
        metrics_df = metrics_df.sort_index()

        # Day-over-day change in ATM IV
        metrics_df["iv_surface_change"] = metrics_df["iv_atm_30d"].diff()

        # Rolling vol-of-vol (std of ATM IV over window)
        metrics_df["iv_vol_of_vol"] = (
            metrics_df["iv_atm_30d"].rolling(_IV_VOL_WINDOW, min_periods=5).std()
        )

        # Reindex to df's dates, forward-fill
        out = df.copy()
        df_dates = pd.to_datetime(df.index.date)
        for col in (
            "iv_atm_30d",
            "iv_skew_25d",
            "iv_term_ratio",
            "iv_surface_change",
            "iv_vol_of_vol",
        ):
            series = metrics_df[col].reindex(df_dates).ffill()
            out[col] = series.values

        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_iv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("iv_atm_30d", "iv_skew_25d", "iv_term_ratio", "iv_surface_change", "iv_vol_of_vol"):
        out[col] = float("nan")
    return out
