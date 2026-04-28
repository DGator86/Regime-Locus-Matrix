"""Factor wrapper for intraday microstructure volume profile metrics."""

from __future__ import annotations

import pandas as pd

from rlm.volume_profile.microstructure_vp import rolling_intraday_vp


class MicrostructureVPFactors:
    """Attach rolling 5-second VP features to a factor frame."""

    def __init__(self, symbol: str, window_seconds: int = 300) -> None:
        self.symbol = symbol
        self.window_seconds = int(window_seconds)

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data.index) == 0:
            return pd.DataFrame(
                columns=["vp_poc", "vp_va_high", "vp_va_low", "vp_hvn_count", "vp_lvn_count"],
                index=data.index,
            )

        vp = rolling_intraday_vp(self.symbol, window_seconds=self.window_seconds)
        if vp.empty:
            out = pd.DataFrame(index=data.index)
            out["vp_poc"] = float("nan")
            out["vp_va_high"] = float("nan")
            out["vp_va_low"] = float("nan")
            out["vp_hvn_count"] = 0
            out["vp_lvn_count"] = 0
            return out

        vp = vp.set_index(pd.to_datetime(vp["timestamp"], utc=True, errors="coerce")).drop(columns=["timestamp"])
        data_idx = pd.DatetimeIndex(pd.to_datetime(data.index, utc=True, errors="coerce"))
        aligned = vp.reindex(data_idx, method="ffill")
        aligned.index = data.index
        return aligned[["vp_poc", "vp_va_high", "vp_va_low", "vp_hvn_count", "vp_lvn_count"]]
