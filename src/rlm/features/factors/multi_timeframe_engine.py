from __future__ import annotations

from dataclasses import replace

import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorSpec


class MultiTimeframeEngine(FactorCalculator):
    """Wrap a factor calculator and project each factor across multiple timeframes.

    For every base factor ``feature`` and timeframe label ``tf``, this wrapper emits:
      - ``mtf_{tf}{feature}``: timeframe-aligned value via ``merge_asof``
      - ``mtf_confluence{feature}``: mean across all generated MTF columns for that feature
    """

    DEFAULT_TIMEFRAMES: tuple[str, ...] = ("15min", "1h", "4h", "1d")

    def __init__(
        self,
        calculator: FactorCalculator,
        *,
        timeframes: tuple[str, ...] | None = None,
    ) -> None:
        self.calculator = calculator
        self.timeframes = timeframes or self.DEFAULT_TIMEFRAMES

    def specs(self) -> list[FactorSpec]:
        base_specs = self.calculator.specs()
        out_specs = list(base_specs)
        for spec in base_specs:
            mtf_names = [self._mtf_name(spec.name, tf) for tf in self.timeframes]
            out_specs.extend(replace(spec, name=mtf_name) for mtf_name in mtf_names)
            out_specs.append(replace(spec, name=self._confluence_name(spec.name)))
        return out_specs

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self.calculator.compute(df)
        if base.empty:
            return base

        time_col = self._resolve_time_column(df)
        aligned_base = base.copy()
        aligned_base["__mtf_time"] = time_col
        aligned_base = aligned_base.sort_values("__mtf_time")
        sorted_index = aligned_base.index

        out = base.copy()
        for feature in base.columns:
            mtf_cols: list[str] = []
            for tf in self.timeframes:
                # Pandas 4: day offset must be uppercase 'D'; hour/min stay lowercase
                if tf[-1] == "d" and (len(tf) == 1 or tf[:-1].isdigit()):
                    tf = tf[:-1] + "D"
                tf_delta = pd.Timedelta(tf)
                tf_label = self._sanitize_tf(tf)
                mtf_col = self._mtf_name(feature, tf)
                mtf_cols.append(mtf_col)

                tf_frame = (
                    aligned_base[["__mtf_time", feature]]
                    .set_index("__mtf_time")
                    .resample(tf)
                    .mean()
                    .dropna(how="all")
                    .reset_index()
                    .rename(columns={feature: mtf_col})
                )

                merged = pd.merge_asof(
                    aligned_base[["__mtf_time"]],
                    tf_frame,
                    on="__mtf_time",
                    direction="backward",
                    tolerance=tf_delta,
                )
                out.loc[sorted_index, mtf_col] = merged[mtf_col].to_numpy()

            out[self._confluence_name(feature)] = out[mtf_cols].mean(axis=1, skipna=True)

        return out

    def _resolve_time_column(self, df: pd.DataFrame) -> pd.Series:
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(df.index, index=df.index)

        for col in ("timestamp", "datetime", "ts", "date"):
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().any():
                    return parsed

        raise ValueError(
            "MultiTimeframeEngine requires a DatetimeIndex or one of: " "timestamp/datetime/ts/date"
        )

    def _mtf_name(self, feature: str, timeframe: str) -> str:
        return f"mtf_{self._sanitize_tf(timeframe)}{feature}"

    def _confluence_name(self, feature: str) -> str:
        return f"mtf_confluence{feature}"

    def _sanitize_tf(self, timeframe: str) -> str:
        clean = "".join(ch for ch in timeframe if ch.isalnum())
        return f"{clean}_"
