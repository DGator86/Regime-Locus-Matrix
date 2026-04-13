from __future__ import annotations

# Completes the full Iceberg liquidity model (Price Action above water + Order Flow below water).
import numpy as np
import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class AdvancedOrderFlowLiquidityFactors(FactorCalculator):
    """Hidden order-flow/liquidity layer factors inspired by SMC/ICT iceberg concepts."""

    def __init__(self) -> None:
        self._specs = [
            FactorSpec(
                name="liquidity_wall",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
            ),
            FactorSpec(
                name="stacked_liquidity_wall",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="liquidity_cloud",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
            ),
            FactorSpec(
                name="liquidity_migration",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.8,
            ),
            FactorSpec(
                name="liquidity_withdrawal",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="micro_void",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.8,
            ),
            FactorSpec(
                name="bsl_detected",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.1,
            ),
            FactorSpec(
                name="ssl_detected",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.1,
            ),
            FactorSpec(
                name="orderflow_confluence_score",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.2,
            ),
        ]

    def specs(self) -> list[FactorSpec]:
        return self._specs

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        volume = (
            df["volume"].astype(float)
            if "volume" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=5).mean().replace(0.0, np.nan)

        rel_volume = volume / volume.rolling(20, min_periods=5).mean().replace(0.0, np.nan)
        rel_volume = rel_volume.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        tol = (atr * 0.15).fillna((high - low).rolling(14, min_periods=5).mean() * 0.1)
        tol = tol.fillna((high - low).replace(0.0, np.nan).median()).clip(lower=1e-8)

        # Equal-high / equal-low cluster density over a short history window.
        near_equal_high = (high - high.rolling(8, min_periods=3).max()).abs() <= tol
        near_equal_low = (low - low.rolling(8, min_periods=3).min()).abs() <= tol
        high_cluster_touches = near_equal_high.rolling(8, min_periods=3).sum()
        low_cluster_touches = near_equal_low.rolling(8, min_periods=3).sum()

        high_side_flow = rel_volume.where(close < open_, 0.0).rolling(8, min_periods=3).sum()
        low_side_flow = rel_volume.where(close > open_, 0.0).rolling(8, min_periods=3).sum()

        wall_strength = ((high_cluster_touches >= 2.0).astype(float) * high_side_flow) + (
            (low_cluster_touches >= 2.0).astype(float) * low_side_flow
        )
        out["liquidity_wall"] = 1.0 + wall_strength.clip(lower=0.0).fillna(0.0) * 0.12

        stacked_strength = (
            (high_cluster_touches >= 3.0).astype(float)
            * (high_cluster_touches - 2.0).clip(lower=0.0)
        ) + (
            (low_cluster_touches >= 3.0).astype(float) * (low_cluster_touches - 2.0).clip(lower=0.0)
        )
        out["stacked_liquidity_wall"] = (
            1.0
            + (stacked_strength * rel_volume.rolling(5, min_periods=2).mean())
            .clip(lower=0.0)
            .fillna(0.0)
            * 0.1
        )

        # Liquidity cloud: high-volume node area with repeated touches near rolling VWAP.
        vwap_num = (close * volume.fillna(0.0)).rolling(30, min_periods=8).sum()
        vwap_den = volume.fillna(0.0).rolling(30, min_periods=8).sum().replace(0.0, np.nan)
        rolling_vwap = (vwap_num / vwap_den).replace([np.inf, -np.inf], np.nan)
        near_vwap = (close - rolling_vwap).abs() <= (tol * 1.5)
        cloud_touches = near_vwap.rolling(30, min_periods=8).sum()
        cloud_volume = rel_volume.rolling(30, min_periods=8).mean()
        out["liquidity_cloud"] = 1.0 + (
            ((cloud_touches / 6.0).clip(lower=0.0) * cloud_volume).clip(lower=0.0).fillna(0.0) * 0.1
        )

        # Migration: directional drift of high-volume node proxy (rolling VWAP slope).
        vwap_shift = rolling_vwap - rolling_vwap.shift(8)
        migration_strength = (vwap_shift.abs() / atr).replace([np.inf, -np.inf], np.nan)
        migration_directional = (migration_strength * np.sign(vwap_shift.fillna(0.0))).abs()
        out["liquidity_migration"] = 1.0 + migration_directional.clip(lower=0.0).fillna(0.0) * 0.25

        # Withdrawal: post-sweep volume decay.
        prior_high_level = high.rolling(8, min_periods=3).max().shift(1)
        prior_low_level = low.rolling(8, min_periods=3).min().shift(1)
        sweep_up = high > (prior_high_level + tol)
        sweep_down = low < (prior_low_level - tol)
        prior_sweep = (sweep_up | sweep_down).shift(1).fillna(False)
        withdrawal_signal = prior_sweep & (rel_volume < 0.75)
        out["liquidity_withdrawal"] = 1.0 + (
            ((0.75 - rel_volume).clip(lower=0.0) * withdrawal_signal.astype(float)).fillna(0.0)
        )

        # Micro-void: compact 1-2 candle imbalance gaps in local structure.
        bullish_micro_gap = (low > high.shift(1)) & ((low - high.shift(1)) <= (tol * 1.2))
        bearish_micro_gap = (high < low.shift(1)) & ((low.shift(1) - high) <= (tol * 1.2))
        micro_void_raw = (
            ((low - high.shift(1)).clip(lower=0.0) + (low.shift(1) - high).clip(lower=0.0)) / atr
        ).replace([np.inf, -np.inf], np.nan)
        micro_void_flag = bullish_micro_gap | bearish_micro_gap
        out["micro_void"] = 1.0 + (micro_void_raw.fillna(0.0) * micro_void_flag.astype(float)).clip(
            lower=0.0
        )

        # BSL/SSL grabs: raid equal highs/lows then reject with strong relative volume.
        rejection_down = close < close.shift(1)
        rejection_up = close > close.shift(1)
        bsl_grab = sweep_up & rejection_down & (rel_volume > 1.2) & (high_cluster_touches >= 2.0)
        ssl_grab = sweep_down & rejection_up & (rel_volume > 1.2) & (low_cluster_touches >= 2.0)

        out["bsl_detected"] = 1.0 + (bsl_grab.astype(float) * (rel_volume - 1.0).clip(lower=0.0))
        out["ssl_detected"] = 1.0 + (ssl_grab.astype(float) * (rel_volume - 1.0).clip(lower=0.0))

        # 0-4 orderflow confluence composite, centered to ratio-style neutral 1.0.
        composite = (
            (out["liquidity_wall"] > 1.05).astype(float)
            + (out["stacked_liquidity_wall"] > 1.03).astype(float)
            + (out["liquidity_cloud"] > 1.05).astype(float)
            + (out["liquidity_migration"] > 1.05).astype(float)
            + (out["liquidity_withdrawal"] > 1.0).astype(float)
            + (out["micro_void"] > 1.0).astype(float)
            + (out["bsl_detected"] > 1.0).astype(float)
            + (out["ssl_detected"] > 1.0).astype(float)
        )
        out["orderflow_confluence_score"] = 1.0 + composite.clip(upper=4.0).fillna(0.0)

        return out[[spec.name for spec in self._specs]]
