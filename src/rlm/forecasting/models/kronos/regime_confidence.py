"""Kronos-driven regime agreement / transition hints (no HuggingFace download in unit tests)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from rlm.forecasting.kronos_config import KronosConfig
from rlm.forecasting.models.kronos.predictor import RLMKronosPredictor

logger = logging.getLogger(__name__)

_MIN_ROWS = 30


def _direction_proxy(last_close: float, closes: np.ndarray) -> float:
    if last_close == 0.0:
        return 0.0
    return float(np.mean(closes) - last_close) / last_close


def _volatility_proxy(last_close: float, highs: np.ndarray, lows: np.ndarray) -> float:
    if last_close <= 0.0:
        return 0.0
    return float(np.mean(highs - lows) / last_close)


def _classify_path(last_close: float, path: np.ndarray) -> str:
    closes = path[:, 3]
    highs = path[:, 1]
    lows = path[:, 2]
    vols = path[:, 4]
    d = _direction_proxy(last_close, closes)
    v = _volatility_proxy(last_close, highs, lows)
    dir_lab = "bull" if d > 0.002 else ("bear" if d < -0.002 else "range")
    vol_lab = "high_vol" if v > 0.02 else "low_vol"
    liq_lab = "high_liquidity" if float(np.mean(vols)) > 140_000 else "low_liquidity"
    trans = "transition" if v > 0.025 else "trend"
    flow = "destabilizing" if abs(d) > 0.01 else "stabilizing"
    return f"{dir_lab}|{trans}|{liq_lab}|{flow}"


class KronosRegimeConfidence:
    """Scores how well Kronos sample paths agree with the current RLM regime_key."""

    def __init__(
        self,
        config: KronosConfig | None = None,
        predictor: Any | None = None,
    ) -> None:
        self.config = config or KronosConfig.from_yaml()
        if predictor is None:
            self._predictor = RLMKronosPredictor(self.config)
        elif isinstance(predictor, RLMKronosPredictor):
            self._predictor = predictor
        else:
            self._predictor = RLMKronosPredictor(self.config, predictor=predictor)

    def score_bar(self, bars: pd.DataFrame, current_regime_key: str | None = None) -> dict[str, Any]:
        if len(bars) < _MIN_ROWS:
            raise ValueError("bars must contain at least 30 rows for KronosRegimeConfidence.score_bar")
        ctx = bars.iloc[-_MIN_ROWS:]
        last_close = float(ctx["close"].iloc[-1])
        try:
            paths = self._predictor.predict_paths(ctx)
        except Exception as exc:
            logger.warning("KronosRegimeConfidence.score_bar predict_paths failed: %s", exc)
            raise
        arr = np.asarray(paths, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"predict_paths must return a 3D array, got shape {arr.shape}")

        per_key = [_classify_path(last_close, arr[i]) for i in range(arr.shape[0])]
        pred_regime = max(set(per_key), key=per_key.count)

        mean_path = np.mean(arr, axis=0)
        fc_ret = (float(mean_path[-1, 3]) - last_close) / last_close
        per_ret = (arr[:, -1, 3] - last_close) / last_close
        fc_vol = float(np.std(per_ret)) if arr.shape[0] > 1 else float(abs(fc_ret))

        base_conf = float(np.clip(0.5 + 0.5 * (1.0 - min(fc_vol * 10.0, 1.0)), 0.0, 1.0))
        epistemic_uncertainty = float(np.clip(1.0 / np.sqrt(max(arr.shape[0], 1)), 0.0, 1.0))
        aleatoric_uncertainty = float(np.clip(min(fc_vol * 10.0, 1.0), 0.0, 1.0))

        if current_regime_key is None:
            agreement = base_conf
            transition = False
            confidence = base_conf
        else:
            matches = sum(1 for k in per_key if k == current_regime_key)
            agreement = matches / max(len(per_key), 1)
            transition = pred_regime != current_regime_key
            confidence = float(
                np.clip(
                    self.config.regime_confidence_weight * agreement
                    + self.config.hmm_confidence_weight * (1.0 - min(fc_vol * 8.0, 1.0)),
                    0.0,
                    1.0,
                )
            )

        return {
            "kronos_confidence": confidence,
            "kronos_epistemic_uncertainty": epistemic_uncertainty,
            "kronos_aleatoric_uncertainty": aleatoric_uncertainty,
            "kronos_regime_agreement": agreement if current_regime_key is not None else confidence,
            "kronos_predicted_regime": pred_regime,
            "kronos_transition_flag": transition,
            "kronos_forecast_return": fc_ret,
            "kronos_forecast_vol": fc_vol,
        }

    def annotate(self, df: pd.DataFrame, min_lookback: int = 30) -> pd.DataFrame:
        if len(df) <= min_lookback:
            return df
        try:
            self._predictor.predict_paths(df.iloc[: min_lookback + 1])
        except ImportError:
            logger.info("KronosRegimeConfidence: no predict_paths backend; skipping overlay")
            return df

        out = df.copy()
        extra = {
            "kronos_confidence": np.nan,
            "kronos_epistemic_uncertainty": np.nan,
            "kronos_aleatoric_uncertainty": np.nan,
            "kronos_regime_agreement": np.nan,
            "kronos_predicted_regime": None,
            "kronos_transition_flag": False,
            "kronos_forecast_return": np.nan,
            "kronos_forecast_vol": np.nan,
        }
        for col, default in extra.items():
            if col not in out.columns:
                out[col] = default
        out["kronos_predicted_regime"] = out["kronos_predicted_regime"].astype(object)

        n = len(out)

        for i in range(min_lookback, n):
            window = df.iloc[: i + 1]
            rk = None
            if "regime_key" in df.columns:
                raw = df["regime_key"].iloc[i]
                rk = None if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw)
            try:
                scored = self.score_bar(window, current_regime_key=rk)
            except Exception:
                continue
            idx = df.index[i]
            for k, v in scored.items():
                out.at[idx, k] = v
        return out
