"""Factor wrapper for cumulative Wyckoff effort/result divergence."""

from __future__ import annotations

import pandas as pd

from rlm.volume_profile.cumulative_wyckoff import session_cumulative_divergence


class CumulativeWyckoffFactors:
    """Produce cumulative Wyckoff divergence score as a factor column."""

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame(columns=["cumulative_wyckoff_score"], index=data.index)

        work = data.copy()
        if "timestamp" not in work.columns:
            work["timestamp"] = pd.to_datetime(work.index, utc=True, errors="coerce")
        start = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").min()
        score = session_cumulative_divergence(work, start.to_pydatetime())
        out = pd.DataFrame(index=data.index)
        out["cumulative_wyckoff_score"] = score.reindex(data.index).fillna(0.0)
        return out
