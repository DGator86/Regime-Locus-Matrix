from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class AdvancedLiquidityPoolFactors(FactorCalculator):
    """Advanced SMC/ICT-style liquidity and imbalance factors."""

    def __init__(self) -> None:
        self._specs = [
            FactorSpec(
                name="liquidity_pool_above",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="liquidity_pool_below",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="liquidity_sweep_confirmed",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.1,
            ),
            FactorSpec(
                name="fvg_bullish",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
            ),
            FactorSpec(
                name="fvg_bearish",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
            ),
            FactorSpec(
                name="order_block_bullish",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="order_block_bearish",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="volume_profile_node_strength",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.8,
            ),
            FactorSpec(
                name="pool_confluence_score",
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

        # Swing detection (fractal-like, 2 candles each side).
        swing_high = (
            high.gt(high.shift(1))
            & high.ge(high.shift(2))
            & high.gt(high.shift(-1))
            & high.ge(high.shift(-2))
        )
        swing_low = (
            low.lt(low.shift(1))
            & low.le(low.shift(2))
            & low.lt(low.shift(-1))
            & low.le(low.shift(-2))
        )

        last_swing_high = high.where(swing_high).ffill()
        last_swing_low = low.where(swing_low).ffill()

        dist_above = (last_swing_high - close).clip(lower=0.0)
        dist_below = (close - last_swing_low).clip(lower=0.0)
        out["liquidity_pool_above"] = 1.0 + (dist_above / atr).clip(lower=0.0).fillna(0.0)
        out["liquidity_pool_below"] = 1.0 + (dist_below / atr).clip(lower=0.0).fillna(0.0)

        # Sweep confirmation: wick sweeps prior swing, close rejects back through level.
        sweep_high = high.gt(last_swing_high.shift(1)) & close.lt(last_swing_high.shift(1))
        sweep_low = low.lt(last_swing_low.shift(1)) & close.gt(last_swing_low.shift(1))
        out["liquidity_sweep_confirmed"] = 1.0 + (sweep_high | sweep_low).astype(float)

        # 3-candle FVG detection (ICT style)
        bullish_gap = (low.shift(-1) - high.shift(1)).clip(lower=0.0)
        bearish_gap = (low.shift(1) - high.shift(-1)).clip(lower=0.0)
        out["fvg_bullish"] = 1.0 + (bullish_gap / atr).clip(lower=0.0).fillna(0.0)
        out["fvg_bearish"] = 1.0 + (bearish_gap / atr).clip(lower=0.0).fillna(0.0)

        # Displacement-based order blocks.
        body = (close - open_).abs()
        body_avg = body.rolling(20, min_periods=5).mean().replace(0.0, np.nan)
        displacement = body / body_avg
        displacement_up = (close > open_) & (displacement > 1.5)
        displacement_down = (close < open_) & (displacement > 1.5)

        bearish_prev_body = (open_.shift(1) - close.shift(1)).clip(lower=0.0)
        bullish_prev_body = (close.shift(1) - open_.shift(1)).clip(lower=0.0)

        bullish_ob = displacement_up & (close.shift(1) < open_.shift(1))
        bearish_ob = displacement_down & (close.shift(1) > open_.shift(1))

        out["order_block_bullish"] = 1.0 + (
            bullish_ob.astype(float) * (bearish_prev_body / atr).fillna(0.0)
        )
        out["order_block_bearish"] = 1.0 + (
            bearish_ob.astype(float) * (bullish_prev_body / atr).fillna(0.0)
        )

        # Simple rolling volume profile node strength using price-bin assignment.
        typical_price = ((high + low + close) / 3.0).replace([np.inf, -np.inf], np.nan)
        vol_profile_strength = pd.Series(1.0, index=df.index, dtype=float)

        bins = 20
        window = 50
        for i in range(len(df)):
            start = max(0, i - window + 1)
            tp_win = typical_price.iloc[start : i + 1]
            vol_win = volume.iloc[start : i + 1]
            valid_win = tp_win.notna() & vol_win.notna() & (vol_win > 0.0)
            if valid_win.sum() < 10:
                continue

            tp_vals = tp_win[valid_win]
            vol_vals = vol_win[valid_win]
            pmin = float(tp_vals.min())
            pmax = float(tp_vals.max())
            if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
                continue

            edges = np.linspace(pmin, pmax, bins + 1)
            bucket = np.digitize(tp_vals.to_numpy(), edges, right=False) - 1
            bucket = np.clip(bucket, 0, bins - 1)

            node_volume = np.bincount(bucket, weights=vol_vals.to_numpy(), minlength=bins)
            if np.all(node_volume <= 0.0):
                continue

            cur_px = typical_price.iloc[i]
            if not np.isfinite(cur_px):
                continue

            cur_bucket = int(np.clip(np.digitize(cur_px, edges, right=False) - 1, 0, bins - 1))
            med_node = float(np.nanmedian(node_volume[node_volume > 0.0]))
            if not np.isfinite(med_node) or med_node <= 0.0:
                continue

            vol_profile_strength.iloc[i] = 1.0 + (float(node_volume[cur_bucket]) / med_node - 1.0)

        out["volume_profile_node_strength"] = vol_profile_strength.replace(0.0, np.nan).fillna(1.0)

        # Confluence score as a positive ratio-style composite centered at 1.0.
        confluence = (
            0.20 * ((out["liquidity_pool_above"] - 1.0) + (out["liquidity_pool_below"] - 1.0))
            + 0.20 * (out["liquidity_sweep_confirmed"] - 1.0)
            + 0.20 * ((out["fvg_bullish"] - 1.0) + (out["fvg_bearish"] - 1.0))
            + 0.20 * ((out["order_block_bullish"] - 1.0) + (out["order_block_bearish"] - 1.0))
            + 0.20 * (out["volume_profile_node_strength"] - 1.0)
        )
        out["pool_confluence_score"] = 1.0 + confluence.clip(lower=0.0).fillna(0.0)

        return out
