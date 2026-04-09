from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from rlm.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class MultiTimeframeLiquidityFactors(FactorCalculator):
    """
    Join pre-computed higher-timeframe (HTF) liquidity features onto the base frame
    and aggregate them into multi-timeframe liquidity signals.

    Expected HTF parquet schema (flexible column matching):
      - timestamp column (or datetime-like index)
      - liquidity bias metric
      - pool confluence metric
      - sweep alignment metric
      - setup strength metric
      - optional alignment score metric

    Missing files are handled gracefully (outputs become NaN).
    """

    _TIMEFRAME_WEIGHTS: Mapping[str, float] = {
        "15m": 0.40,
        "1h": 0.30,
        "4h": 0.20,
        "1d": 0.10,
    }

    _METRICS: tuple[str, ...] = (
        "liquidity_bias",
        "pool_confluence",
        "sweep_aligned",
        "liquidity_setup_strength",
        "alignment_score",
    )

    def __init__(
        self,
        *,
        htf_parquet_paths: Mapping[str, str | Path] | None = None,
        merge_tolerance: pd.Timedelta | str | None = "7D",
    ) -> None:
        default_root = Path("data/features/liquidity")
        self._paths: dict[str, Path] = {
            tf: Path(default_root / f"liquidity_{tf}.parquet") for tf in self._TIMEFRAME_WEIGHTS
        }
        if htf_parquet_paths:
            self._paths.update({tf: Path(path) for tf, path in htf_parquet_paths.items()})

        self._merge_tolerance = pd.Timedelta(merge_tolerance) if merge_tolerance else None
        self._htf_cache: dict[str, pd.DataFrame | None] = {}

        self._specs = [
            FactorSpec(
                name="htf_liquidity_bias",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="mtf_pool_confluence",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="htf_sweep_aligned",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="mtf_liquidity_setup_strength",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="tf_alignment_score",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.5,
                k=1.0,
            ),
        ]

    def specs(self) -> list[FactorSpec]:
        return self._specs

    @staticmethod
    def _as_timestamp_frame(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(df.index, utc=True, errors="coerce")

        out = df.copy()
        out["__timestamp"] = ts
        out = out.dropna(subset=["__timestamp"]).sort_values("__timestamp")
        return out

    @staticmethod
    def _candidate_columns(tf: str, metric: str) -> tuple[str, ...]:
        return (
            f"{tf}_{metric}",
            f"htf_{metric}",
            metric,
            f"liquidity_{metric}",
        )

    def _load_htf_frame(self, tf: str) -> pd.DataFrame | None:
        if tf in self._htf_cache:
            return self._htf_cache[tf]

        path = self._paths.get(tf)
        if path is None or not path.exists():
            self._htf_cache[tf] = None
            return None

        htf = pd.read_parquet(path)
        htf = self._as_timestamp_frame(htf)

        selected: dict[str, pd.Series] = {"__timestamp": htf["__timestamp"]}
        for metric in self._METRICS:
            for col in self._candidate_columns(tf=tf, metric=metric):
                if col in htf.columns:
                    selected[f"{tf}__{metric}"] = pd.to_numeric(htf[col], errors="coerce")
                    break

        loaded = pd.DataFrame(selected).sort_values("__timestamp")
        self._htf_cache[tf] = loaded
        return loaded

    @staticmethod
    def _weighted_average(values: dict[str, pd.Series], weights: Mapping[str, float]) -> pd.Series:
        if not values:
            return pd.Series(dtype=float)

        ordered_tfs = [tf for tf in weights if tf in values]
        matrix = pd.concat([values[tf] for tf in ordered_tfs], axis=1)
        w = np.array([weights[tf] for tf in ordered_tfs], dtype=float)

        valid = matrix.notna().to_numpy(dtype=float)
        numer = np.nansum(matrix.to_numpy(dtype=float) * w, axis=1)
        denom = np.sum(valid * w, axis=1)

        out = np.divide(
            numer,
            denom,
            out=np.full_like(numer, fill_value=np.nan, dtype=float),
            where=denom > 0,
        )
        return pd.Series(out, index=matrix.index, dtype=float)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self._as_timestamp_frame(df)

        aligned = base[["__timestamp"]].copy()
        for tf in self._TIMEFRAME_WEIGHTS:
            htf = self._load_htf_frame(tf)
            if htf is None:
                continue
            aligned = pd.merge_asof(
                aligned,
                htf,
                on="__timestamp",
                direction="backward",
                tolerance=self._merge_tolerance,
            )

        per_metric: dict[str, dict[str, pd.Series]] = {m: {} for m in self._METRICS}
        for tf in self._TIMEFRAME_WEIGHTS:
            for metric in self._METRICS:
                col = f"{tf}__{metric}"
                if col in aligned.columns:
                    per_metric[metric][tf] = aligned[col]

        out = pd.DataFrame(index=base.index)
        out["htf_liquidity_bias"] = self._weighted_average(
            per_metric["liquidity_bias"], self._TIMEFRAME_WEIGHTS
        )
        out["mtf_pool_confluence"] = self._weighted_average(
            per_metric["pool_confluence"], self._TIMEFRAME_WEIGHTS
        )
        out["htf_sweep_aligned"] = self._weighted_average(
            per_metric["sweep_aligned"], self._TIMEFRAME_WEIGHTS
        )
        out["mtf_liquidity_setup_strength"] = self._weighted_average(
            per_metric["liquidity_setup_strength"], self._TIMEFRAME_WEIGHTS
        )

        if per_metric["alignment_score"]:
            out["tf_alignment_score"] = self._weighted_average(
                per_metric["alignment_score"], self._TIMEFRAME_WEIGHTS
            )
        elif per_metric["liquidity_bias"]:
            signed_bias = {
                tf: np.sign(series) for tf, series in per_metric["liquidity_bias"].items()
            }
            agreement = self._weighted_average(signed_bias, self._TIMEFRAME_WEIGHTS)
            out["tf_alignment_score"] = agreement.abs()
        else:
            out["tf_alignment_score"] = np.nan

        return out.reindex(df.index)
