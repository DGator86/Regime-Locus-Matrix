"""FactorCalculator that derives direction / volatility factors from Kronos predictions."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rlm.factors.base import FactorCalculator
from rlm.kronos.config import KronosConfig
from rlm.kronos.predictor import RLMKronosPredictor
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind

logger = logging.getLogger(__name__)

_MIN_LOOKBACK = 30


class KronosFactorCalculator(FactorCalculator):
    """Produces Kronos-derived factors for the factor pipeline.

    Factors:
    * ``kronos_return_forecast`` (DIRECTION) -- predicted 1-bar return
    * ``kronos_range_forecast`` (VOLATILITY) -- predicted normalised range
    * ``kronos_path_dispersion`` (VOLATILITY) -- stdev of returns across samples
    """

    def __init__(
        self,
        config: KronosConfig | None = None,
        predictor: RLMKronosPredictor | None = None,
    ) -> None:
        """
        Initialize the KronosFactorCalculator with a configuration and predictor, defaulting to file-based config and a new RLMKronosPredictor when not provided.
        
        Parameters:
            config (KronosConfig | None): Optional configuration to use; when omitted, loaded via KronosConfig.from_yaml().
            predictor (RLMKronosPredictor | None): Optional predictor instance to use; when omitted, a new RLMKronosPredictor is constructed with the resolved config.
        """
        self._config = config or KronosConfig.from_yaml()
        self._predictor = predictor or RLMKronosPredictor(self._config)

    def specs(self) -> list[FactorSpec]:
        """
        Return the FactorSpec list describing the three Kronos-derived factors produced by this calculator.
        
        The returned list defines:
        - `kronos_return_forecast`: signed direction forecast scaled by 0.01 (one-bar return forecast).
        - `kronos_range_forecast`: volatility ratio representing predicted high-low range normalized by price with a neutral value of 0.01.
        - `kronos_path_dispersion`: volatility ratio representing dispersion of predicted sample returns with a neutral value of 0.005.
        
        Returns:
            A list of `FactorSpec` objects for `kronos_return_forecast`, `kronos_range_forecast`, and `kronos_path_dispersion`.
        """
        return [
            FactorSpec(
                name="kronos_return_forecast",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.01,
                k=1.0,
            ),
            FactorSpec(
                name="kronos_range_forecast",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.01,
                k=1.0,
            ),
            FactorSpec(
                name="kronos_path_dispersion",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.005,
                k=1.0,
            ),
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute three Kronos-derived factor series aligned to the input DataFrame's index.
        
        Processes each eligible row of `df` using the instance's predictor to produce:
        - a 1-bar signed return forecast ("kronos_return_forecast"),
        - a normalized predicted high-low range ("kronos_range_forecast"),
        - and a dispersion of per-sample predicted returns ("kronos_path_dispersion").
        Rows with insufficient lookback data, a zero close price, or failed predictions are left as NaN.
        
        Parameters:
            df (pd.DataFrame): Input OHLCV-like DataFrame indexed by timestamp and containing at least a "close" column.
        
        Returns:
            pd.DataFrame: New DataFrame indexed like `df` with columns:
                - "kronos_return_forecast": predicted 1-bar return (signed).
                - "kronos_range_forecast": predicted mean high-minus-low normalized by current close.
                - "kronos_path_dispersion": standard deviation of per-sample predicted returns (0.0 when only one sample).
        """
        n = len(df)
        ret_forecast = np.full(n, np.nan)
        range_forecast = np.full(n, np.nan)
        path_dispersion = np.full(n, np.nan)

        cfg = self._config

        for idx in range(_MIN_LOOKBACK, n):
            lookback_start = max(0, idx - cfg.max_context + 1)
            bar_slice = df.iloc[lookback_start : idx + 1]
            current_close = float(df.iloc[idx]["close"])

            if current_close == 0:
                continue

            try:
                paths = self._predictor.predict_paths(bar_slice)
            except Exception:
                logger.debug("Kronos factor compute failed at idx %d", idx, exc_info=True)
                continue

            # paths: (sample_count, pred_len, 6) -- OHLCV + amount
            mean_path = np.mean(paths, axis=0)  # (pred_len, 6)
            pred_close = mean_path[-1, 3]
            pred_ret = (pred_close - current_close) / current_close
            pred_range = np.mean(mean_path[:, 1] - mean_path[:, 2]) / current_close

            per_sample_returns = (paths[:, -1, 3] - current_close) / current_close
            dispersion = float(np.std(per_sample_returns)) if paths.shape[0] > 1 else 0.0

            ret_forecast[idx] = pred_ret
            range_forecast[idx] = pred_range
            path_dispersion[idx] = dispersion

        out = pd.DataFrame(index=df.index)
        out["kronos_return_forecast"] = ret_forecast
        out["kronos_range_forecast"] = range_forecast
        out["kronos_path_dispersion"] = path_dispersion
        return out
