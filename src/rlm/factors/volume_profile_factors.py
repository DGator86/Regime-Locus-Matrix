"""Volume profile factor calculator integration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from rlm.features.factors.base import FactorCalculator as BaseFactorCalculator
else:
    try:
        from rlm.features.factors.base import FactorCalculator as BaseFactorCalculator
    except Exception:  # pragma: no cover

        class BaseFactorCalculator:
            def compute(self, data: pd.DataFrame) -> pd.DataFrame:
                raise NotImplementedError


from rlm.volume_profile.auction_metrics import (
    auction_state,
    effort_result_divergence,
    value_area_migration,
)
from rlm.volume_profile.cumulative_wyckoff import cumulative_effort_result
from rlm.volume_profile.fx_session_profiles import get_fx_session_profile
from rlm.volume_profile.profile_calculator import calculate_volume_profile, identify_nodes


class VolumeProfileFactors(BaseFactorCalculator):
    """Compute daily/session-oriented volume profile features."""

    def __init__(
        self,
        rolling_window_days: int = 20,
        price_precision: int = 400,
        session_type: str = "equity",
    ) -> None:
        self.rolling_window_days = int(rolling_window_days)
        self.price_precision = int(price_precision)
        self.session_type = str(session_type)

    def _session_dates(self, data: pd.DataFrame) -> Iterable[pd.Timestamp]:
        if "timestamp" in data.columns:
            ts = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(data.index, utc=True, errors="coerce")
        return ts.dt.floor("D")

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["vp_poc"] = np.nan
        out["vp_va_high"] = np.nan
        out["vp_va_low"] = np.nan
        out["vp_hvn_count"] = 0
        out["vp_lvn_count"] = 0
        out["vp_effort_result_score"] = 0.0
        out["vp_auction_state"] = "balance"
        out["vp_va_migration"] = "neutral"
        out["cumulative_wyckoff_score"] = 0.0

        if not {"close", "volume"}.issubset(out.columns):
            return out

        dates = self._session_dates(out)
        work = out.assign(__session_date=dates)
        profiles: list[dict[str, object]] = []

        for session_date, session_df in work.groupby("__session_date"):
            if pd.isna(session_date):
                continue

            session_indices = list(session_df.index)
            n_bars = len(session_indices)
            last_profile: dict[str, object] | None = None

            for j, bar_idx in enumerate(session_indices):
                # Use only bars up to and including the current one (causal).
                partial_df = session_df.iloc[: j + 1]

                profile_input = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            partial_df.get("timestamp", partial_df.index), utc=True, errors="coerce"
                        ),
                        "price": pd.to_numeric(partial_df["close"], errors="coerce"),
                        "volume": pd.to_numeric(partial_df["volume"], errors="coerce"),
                    },
                    index=partial_df.index,
                ).dropna(subset=["timestamp", "price", "volume"])

                if profile_input.empty:
                    continue

                if self.session_type == "fx":
                    profile = get_fx_session_profile(
                        profile_input, "London", pd.Timestamp(session_date).to_pydatetime()
                    )
                else:
                    profile = calculate_volume_profile(profile_input, price_precision=self.price_precision)
                last_profile = profile
                nodes = identify_nodes(profile["volume_profile_series"])
                effort_score = effort_result_divergence(partial_df, profile)
                cum_wyckoff = cumulative_effort_result(
                    pd.DataFrame(
                        {
                            "high": partial_df.get("high", partial_df["close"]),
                            "low": partial_df.get("low", partial_df["close"]),
                            "close": partial_df["close"],
                            "volume": partial_df["volume"],
                        }
                    )
                )

                # Migration uses historical completed sessions + the current partial one.
                historical = profiles[-(self.rolling_window_days - 1) :] + [profile]
                migration = value_area_migration(historical)
                state = auction_state(profile, float(partial_df["close"].iloc[-1]))

                out.at[bar_idx, "vp_poc"] = float(profile["poc"])
                out.at[bar_idx, "vp_va_high"] = float(profile["value_area_high"])
                out.at[bar_idx, "vp_va_low"] = float(profile["value_area_low"])
                out.at[bar_idx, "vp_hvn_count"] = len(nodes["hvn_levels"])
                out.at[bar_idx, "vp_lvn_count"] = len(nodes["lvn_levels"])
                out.at[bar_idx, "vp_effort_result_score"] = float(effort_score)
                out.at[bar_idx, "vp_auction_state"] = state
                out.at[bar_idx, "vp_va_migration"] = migration
                out.at[bar_idx, "cumulative_wyckoff_score"] = float(cum_wyckoff)

            # Append the completed session's final profile for historical migration tracking.
            # The last loop iteration (j == n_bars - 1) already computed profile from the
            # full session, so reuse it directly instead of recomputing.
            if last_profile is not None:
                profiles.append(last_profile)

        return out[
            [
                "vp_poc",
                "vp_va_high",
                "vp_va_low",
                "vp_hvn_count",
                "vp_lvn_count",
                "vp_effort_result_score",
                "vp_auction_state",
                "vp_va_migration",
                "cumulative_wyckoff_score",
            ]
        ]
