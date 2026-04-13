"""Kronos regime-confidence engine.

For each bar, predicts future OHLCV via Kronos, derives lightweight
factor proxies from the predicted bars, classifies the predicted regime,
and compares it to the current regime to produce confidence / agreement
signals that compound with HMM confidence.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rlm.forecasting.models.kronos.config import KronosConfig
from rlm.forecasting.models.kronos.predictor import RLMKronosPredictor
from rlm.features.scoring.state_matrix import make_regime_key
from rlm.features.scoring.thresholds import (
    classify_dealer_flow,
    classify_direction,
    classify_liquidity,
    classify_volatility,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Clamp factor proxies into [-1, 1] to match the tanh-standardised space
# that the scoring thresholds expect.
_CLIP = 1.0


def _direction_proxy(current_close: float, predicted_closes: np.ndarray) -> float:
    """
    Compute a scaled direction proxy from the multi-bar predicted closes, clamped to [-1, 1].
    
    Parameters:
        current_close (float): The most-recent observed close price; if zero, the function returns 0.0.
        predicted_closes (np.ndarray): Array of predicted close prices; the final element is used to compute the multi-bar return.
    
    Returns:
        float: Clamped direction proxy in the range [-1, 1]; positive values indicate an expected upward return, negative values indicate an expected downward return.
    """
    if current_close == 0:
        return 0.0
    final_ret = (predicted_closes[-1] - current_close) / current_close
    # Scale: +-2% maps to roughly +-0.8 so the thresholds (0.3 / 0.6) fire
    return float(np.clip(final_ret * 40.0, -_CLIP, _CLIP))


def _volatility_proxy(current_close: float, predicted_highs: np.ndarray, predicted_lows: np.ndarray) -> float:
    """
    Compute a normalized volatility proxy from predicted highs and lows.
    
    If `current_close` is zero, returns `0.0`. Otherwise computes the mean predicted high-low range
    relative to `current_close`, applies a linear transform (avg_range * 50.0 - 0.5) and clamps the
    result to the interval [-1.0, 1.0].
    
    Parameters:
        current_close (float): Last observed close price used for normalization.
        predicted_highs (np.ndarray): Array of predicted high prices for each future step.
        predicted_lows (np.ndarray): Array of predicted low prices for each future step.
    
    Returns:
        float: A value in [-1.0, 1.0] representing the normalized predicted volatility
        (approximately 0.5 corresponds to a 1% average predicted range before clamping).
    """
    if current_close == 0:
        return 0.0
    avg_range = np.mean(predicted_highs - predicted_lows) / current_close
    # Scale so that a 1% average range is about 0.5
    return float(np.clip(avg_range * 50.0 - 0.5, -_CLIP, _CLIP))


def _classify_path(current_close: float, path: np.ndarray) -> str:
    """
    Assigns a regime key to a single predicted OHLCV path.
    
    Parameters:
        current_close (float): Reference close price for computing proxies.
        path (np.ndarray): Predicted path with shape (pred_len, 6); columns are OHLCV + amount where index 3 is close, 1 is high, and 2 is low.
    
    Returns:
        Regime key string that encodes discrete labels for direction, volatility, liquidity, and dealer-flow axes.
    """
    closes = path[:, 3]
    highs = path[:, 1]
    lows = path[:, 2]

    s_d = _direction_proxy(current_close, closes)
    s_v = _volatility_proxy(current_close, highs, lows)
    # Liquidity and dealer flow cannot be inferred from OHLCV alone;
    # use neutral defaults so those axes don't dominate the regime key.
    s_l = 0.0
    s_g = 0.0

    return make_regime_key(
        classify_direction(s_d),
        classify_volatility(s_v),
        classify_liquidity(s_l),
        classify_dealer_flow(s_g),
    )


class KronosRegimeConfidence:
    """Produces per-bar Kronos regime-confidence columns.

    Typical usage inside a pipeline or backtest loop::

        krc = KronosRegimeConfidence()
        df = krc.annotate(df)
    """

    def __init__(
        self,
        config: KronosConfig | None = None,
        predictor: RLMKronosPredictor | None = None,
    ) -> None:
        """
        Initialize the KronosRegimeConfidence with a configuration and a predictor.
        
        Parameters:
            config (KronosConfig | None): Configuration to use. If `None`, the configuration is loaded from YAML via KronosConfig.from_yaml().
            predictor (RLMKronosPredictor | None): Predictor instance to use for path generation. If `None`, an RLMKronosPredictor is created using the resolved `config`.
        """
        self._config = config or KronosConfig.from_yaml()
        self._predictor = predictor or RLMKronosPredictor(self._config)

    # ------------------------------------------------------------------
    # Per-bar scoring
    # ------------------------------------------------------------------

    def score_bar(
        self,
        bar_df: pd.DataFrame,
        current_regime_key: str | None = None,
    ) -> dict[str, object]:
        """
        Compute per-bar Kronos regime-confidence metrics for the final row in bar_df.
        
        Parameters:
            bar_df (pd.DataFrame): OHLCV history ending at the bar to be scored. Shorter-than-configured context is allowed but may reduce prediction quality.
            current_regime_key (str | None): The current bar's regime key, if available; when provided, agreement and transition metrics are computed relative to this key.
        
        Returns:
            dict: A mapping with the following keys:
                - kronos_confidence: Fraction of predicted sample paths that share the modal predicted regime.
                - kronos_regime_agreement: Fraction of predicted paths matching `current_regime_key` (or equal to `kronos_confidence` if `current_regime_key` is None).
                - kronos_predicted_regime: The modal predicted regime key (string).
                - kronos_transition_flag: `True` if the modal predicted regime differs from `current_regime_key` (False when `current_regime_key` is None).
                - kronos_forecast_return: Mean predicted return (final predicted close vs current close) across samples.
                - kronos_forecast_vol: Standard deviation of predicted returns across samples (0.0 if only one sample).
        """
        current_close = float(bar_df["close"].iloc[-1])

        paths = self._predictor.predict_paths(bar_df)
        # paths: (sample_count, pred_len, 6)

        regime_keys: list[str] = []
        returns: list[float] = []

        for i in range(paths.shape[0]):
            path = paths[i]
            rk = _classify_path(current_close, path)
            regime_keys.append(rk)

            pred_close = path[-1, 3]
            ret = (pred_close - current_close) / current_close if current_close else 0.0
            returns.append(ret)

        counts = Counter(regime_keys)
        mode_regime, mode_count = counts.most_common(1)[0]
        sample_count = len(regime_keys)

        kronos_confidence = mode_count / sample_count

        if current_regime_key is not None:
            agreement_count = sum(1 for rk in regime_keys if rk == current_regime_key)
            kronos_regime_agreement = agreement_count / sample_count
            kronos_transition_flag = mode_regime != current_regime_key
        else:
            kronos_regime_agreement = kronos_confidence
            kronos_transition_flag = False

        returns_arr = np.array(returns)
        kronos_forecast_return = float(np.mean(returns_arr))
        kronos_forecast_vol = float(np.std(returns_arr)) if len(returns_arr) > 1 else 0.0

        return {
            "kronos_confidence": kronos_confidence,
            "kronos_regime_agreement": kronos_regime_agreement,
            "kronos_predicted_regime": mode_regime,
            "kronos_transition_flag": kronos_transition_flag,
            "kronos_forecast_return": kronos_forecast_return,
            "kronos_forecast_vol": kronos_forecast_vol,
        }

    # ------------------------------------------------------------------
    # Batch annotation
    # ------------------------------------------------------------------

    def annotate(
        self,
        df: pd.DataFrame,
        *,
        regime_key_col: str = "regime_key",
        min_lookback: int = 30,
    ) -> pd.DataFrame:
        """
        Add Kronos regime-confidence columns to a copy of the input DataFrame.
        
        Initializes the following columns with neutral defaults for every row:
        `kronos_confidence`, `kronos_regime_agreement`, `kronos_predicted_regime`,
        `kronos_transition_flag`, `kronos_forecast_return`, and `kronos_forecast_vol`.
        Starting at `min_lookback`, computes per-row scores by calling `score_bar`
        on a lookback slice bounded by `self._config.max_context`; if `regime_key_col`
        exists the current row's value is passed as `current_regime_key`. If scoring
        fails for a row, the row's defaults are left unchanged and the error is logged
        at debug level.
        
        Parameters:
            df (pd.DataFrame): Input bars DataFrame to annotate.
            regime_key_col (str): Column name containing the current regime key, if present.
            min_lookback (int): First row index to score; rows with index < this value keep defaults.
        
        Returns:
            pd.DataFrame: A copy of `df` with the Kronos confidence and forecast columns added.
        """
        out = df.copy()

        col_defaults: dict[str, object] = {
            "kronos_confidence": np.nan,
            "kronos_regime_agreement": np.nan,
            "kronos_predicted_regime": "",
            "kronos_transition_flag": False,
            "kronos_forecast_return": np.nan,
            "kronos_forecast_vol": np.nan,
        }
        for col, default in col_defaults.items():
            out[col] = default

        has_regime = regime_key_col in out.columns

        for idx in range(min_lookback, len(out)):
            lookback_start = max(0, idx - self._config.max_context + 1)
            bar_slice = out.iloc[lookback_start : idx + 1]

            current_rk = str(out.iloc[idx][regime_key_col]) if has_regime else None

            try:
                scores = self.score_bar(bar_slice, current_regime_key=current_rk)
                for col, val in scores.items():
                    out.iat[idx, out.columns.get_loc(col)] = val
            except Exception:
                logger.debug("Kronos score_bar failed at index %d", idx, exc_info=True)

        return out
