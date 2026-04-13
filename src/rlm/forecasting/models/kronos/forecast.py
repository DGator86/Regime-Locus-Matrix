"""KronosForecastPipeline -- drop-in ForecastPipeline replacement.

Produces the same column contract (mu, sigma, mean_price, forecast bands)
using multi-sample Kronos predictions instead of distribution estimation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rlm.forecasting.bands import compute_state_matrix_bands
from rlm.forecasting.models.kronos.config import KronosConfig
from rlm.forecasting.models.kronos.predictor import RLMKronosPredictor

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
        """
        Initialize the KronosForecastPipeline with configuration, predictor, and minimum lookback.
        
        Parameters:
            config (KronosConfig | None): Configuration to use; if None, loaded via KronosConfig.from_yaml().
            predictor (RLMKronosPredictor | None): Predictor instance to use; if None, a RLMKronosPredictor is created with `config`.
            min_lookback (int): Minimum number of historical bars required before producing forecasts (start index offset).
        """
        self._config = config or KronosConfig.from_yaml()
        self._predictor = predictor or RLMKronosPredictor(self._config)
        self._min_lookback = min_lookback

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and attach Kronos-based forecast columns to the given DataFrame.
        
        Adds or overwrites per-row forecast fields including `mu`, `sigma`, `mean_price`, `forecast_return`,
        `forecast_return_lower`, `forecast_return_median`, `forecast_return_upper`, `forecast_uncertainty`,
        and `forecast_source`, and also populates state-matrix band columns. Forecasts are produced from
        multi-sample Kronos prediction paths; for each evaluated row the final predicted close per sample is
        converted to sample returns and used to derive mean (`mu`), standard deviation (`sigma`, lower-bounded
        to 1e-4), and the 10th/50th/90th return quantiles (lower/median/upper). Rows with insufficient lookback,
        a zero `close`, or where prediction fails remain at their initialized default values.
        
        Returns:
            pd.DataFrame: A copy of the input DataFrame with the added forecast and band columns.
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
