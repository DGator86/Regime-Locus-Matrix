from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from rlm.factors.pipeline import FactorPipeline


def parse_higher_tfs(raw: str) -> tuple[str, ...]:
    parts = [p.strip() for p in str(raw).split(",")]
    values = tuple(p for p in parts if p)
    if not values:
        raise ValueError("Expected at least one higher timeframe (example: 1W,1M).")
    return values


@dataclass(frozen=True)
class MultiTimeframeEngine:
    higher_tfs: tuple[str, ...] = ("1W", "1M")

    def _suffix_for_tf(self, tf: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", tf.lower()).strip("_")

    def _resample_bars(self, bars: pd.DataFrame, tf: str) -> pd.DataFrame:
        agg: dict[str, str] = {}
        if "open" in bars.columns:
            agg["open"] = "first"
        if "high" in bars.columns:
            agg["high"] = "max"
        if "low" in bars.columns:
            agg["low"] = "min"
        if "close" in bars.columns:
            agg["close"] = "last"
        if "volume" in bars.columns:
            agg["volume"] = "sum"
        for col in bars.columns:
            if col not in agg and pd.api.types.is_numeric_dtype(bars[col]):
                agg[col] = "last"
        if not agg:
            return bars.iloc[0:0].copy()
        out = bars.resample(tf).agg(agg).dropna(subset=[c for c in ("open", "high", "low", "close") if c in agg])
        return out

    def augment_factors(self, bars: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        out = factors.copy()
        for tf in self.higher_tfs:
            htf_bars = self._resample_bars(bars, tf)
            if htf_bars.empty:
                continue
            htf_factors = FactorPipeline().run(htf_bars)
            keep_cols = [c for c in ("S_D", "S_V", "S_L", "S_G") if c in htf_factors.columns]
            if not keep_cols:
                continue
            suffix = self._suffix_for_tf(tf)
            expanded = htf_factors[keep_cols].reindex(out.index, method="ffill").add_prefix(f"htf_{suffix}_")
            out = out.join(expanded)
        return out


def format_precompute_instructions(*, symbol: str, higher_tfs: tuple[str, ...]) -> str:
    tf_str = ",".join(higher_tfs)
    return (
        "Pre-compute HTF factors before heavy runs:\n"
        f"  python3 scripts/run_forecast_pipeline.py --symbol {symbol} --mtf --higher-tfs {tf_str} --no-vix\n"
        f"  python3 scripts/run_backtest.py --symbol {symbol} --mtf --higher-tfs {tf_str} --no-vix"
    )
