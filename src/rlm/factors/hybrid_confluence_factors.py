"""Factor wrapper for VP/GEX/IV hybrid confluence metrics."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from rlm.volume_profile.hybrid_confluence import hybrid_support_resistance


class HybridConfluenceFactors:
    """Derive scalar confluence features from hybrid level analysis."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["vp_gex_confluence_poc"] = float("nan")
        out["vp_iv_skew_poc"] = float("nan")
        out["vp_hybrid_strength_max"] = float("nan")
        if data.empty:
            return out

        last = data.iloc[-1]
        ts = last.get("timestamp", datetime.utcnow())
        vp_profile = {
            "poc": last.get("vp_poc", float("nan")),
            "value_area_high": last.get("vp_va_high", float("nan")),
            "value_area_low": last.get("vp_va_low", float("nan")),
            "hvn_levels": [],
            "lvn_levels": [],
        }
        levels = hybrid_support_resistance(
            self.symbol, pd.Timestamp(ts).to_pydatetime(), vp_profile
        )
        if levels.empty:
            return out

        poc_row = levels.iloc[(levels["level"] - float(vp_profile["poc"])).abs().argsort().iloc[0]]
        out.loc[:, "vp_gex_confluence_poc"] = float(poc_row.get("gex_percentile", float("nan")))
        out.loc[:, "vp_iv_skew_poc"] = float(poc_row.get("iv_skew", float("nan")))
        out.loc[:, "vp_hybrid_strength_max"] = float(levels["strength_score"].max())
        return out
