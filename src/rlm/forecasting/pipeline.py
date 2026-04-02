from __future__ import annotations

import pandas as pd

from rlm.forecasting.bands import compute_state_matrix_bands
from rlm.forecasting.distribution import estimate_distribution
from rlm.types.forecast import ForecastConfig


class ForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        move_window: int = 100,
        vol_window: int = 100,
    ) -> None:
        self.config = config or ForecastConfig()
        self.move_window = move_window
        self.vol_window = vol_window

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = estimate_distribution(
            df=df,
            config=self.config,
            move_window=self.move_window,
            vol_window=self.vol_window,
        )
        out = compute_state_matrix_bands(out)
        return out
