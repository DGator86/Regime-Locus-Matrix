"""KronosForecastPipeline -- drop-in ForecastPipeline replacement.

Produces the same column contract (mu, sigma, mean_price, forecast bands)
using multi-sample Kronos predictions instead of distribution estimation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rlm.forecasting.bands import compute_state_matrix_bands
from rlm.kronos.config import KronosConfig
from rlm.kronos.predictor import RLMKronosPredictor

logger = logging.getLogger(__name__)


class KronosForecastPipeline:
    """Forecast ``mu`` / ``sigma`` / ``mean_price`` from Kronos sample paths.

    Compatible with the column contract of
    :class:`rlm.forecasting.pipeline.ForecastPipeline` so it can serve as
    the inner engine of ``HybridForecastPipeline``.
    """

    def __init__(
        self,
        config: KronosConfig | None = None,
        predictor: RLMKronosPredictor | None = None,
        min_lookback: int = 30,
    ) -> None:
        self._config = config or KronosConfig.from_yaml()
        self._predictor = predictor or RLMKronosPredictor(self._config)
        self._min_lookback = min_lookback

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forecast columns to *df* using Kronos predictions.

        The following columns are added / overwritten:

        * ``mu``, ``sigma`` -- return-scale drift and vol
        * ``mean_price`` -- ``close * (1 + mu)``
        * ``forecast_return``, ``forecast_return_lower``,
          ``forecast_return_median``, ``forecast_return_upper``
        * ``forecast_uncertainty``, ``forecast_source``
        * Band columns via :func:`compute_state_matrix_bands`
        """
        out = df.copy()

        n = len(out)
        out["mu"] = 0.0
        out["sigma"] = 1e-4
        out["mean_price"] = out["close"].copy()
        out["forecast_return"] = 0.0
        out["forecast_return_lower"] = 0.0
        out["forecast_return_median"] = 0.0
        out["forecast_return_upper"] = 0.0
        out["forecast_uncertainty"] = 0.0
        out["forecast_source"] = "kronos"

        cfg = self._config

        for idx in range(self._min_lookback, n):
            lookback_start = max(0, idx - cfg.max_context + 1)
            bar_slice = out.iloc[lookback_start : idx + 1]
            current_close = float(out.iloc[idx]["close"])

            if current_close == 0:
                continue

            try:
                paths = self._predictor.predict_paths(bar_slice)
            except Exception:
                logger.debug("Kronos forecast failed at index %d", idx, exc_info=True)
                continue

            # paths: (sample_count, pred_len, 6)  -- cols: OHLCV + amount
            pred_closes = paths[:, -1, 3]  # final predicted close per sample
            returns = (pred_closes - current_close) / current_close

            mu = float(np.mean(returns))
            sigma = max(float(np.std(returns)), 1e-4)

            lower = float(np.percentile(returns, 10))
            median = float(np.median(returns))
            upper = float(np.percentile(returns, 90))

            out.iat[idx, out.columns.get_loc("mu")] = mu
            out.iat[idx, out.columns.get_loc("sigma")] = sigma
            out.iat[idx, out.columns.get_loc("mean_price")] = current_close * (1 + mu)
            out.iat[idx, out.columns.get_loc("forecast_return")] = mu
            out.iat[idx, out.columns.get_loc("forecast_return_lower")] = lower
            out.iat[idx, out.columns.get_loc("forecast_return_median")] = median
            out.iat[idx, out.columns.get_loc("forecast_return_upper")] = upper
            out.iat[idx, out.columns.get_loc("forecast_uncertainty")] = upper - lower

        finetuned = cfg.finetuned_model_path is not None
        out["forecast_source"] = "kronos_finetuned" if finetuned else "kronos"

        out = compute_state_matrix_bands(out)
        return out
