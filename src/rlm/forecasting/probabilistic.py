from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from rlm.forecasting.bands import compute_state_matrix_bands
from rlm.forecasting.distribution import estimate_distribution
from rlm.types.forecast import ForecastConfig

DEFAULT_PROB_FEATURE_COLUMNS: tuple[str, ...] = (
    "S_D",
    "S_V",
    "S_L",
    "S_G",
    "b_m",
    "b_sigma",
    "mu",
    "sigma",
    "realized_vol",
    "bid_ask_spread",
    "term_structure_ratio",
    "iv_rank",
    "put_call_skew",
    "vix",
    "vvix",
)


@dataclass(frozen=True)
class QuantileLinearModelArtifact:
    quantiles: tuple[float, ...]
    feature_columns: tuple[str, ...]
    intercepts: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]

    @classmethod
    def load(cls, path: str | Path) -> "QuantileLinearModelArtifact":
        payload = json.loads(Path(path).read_text())
        return cls(
            quantiles=tuple(float(x) for x in payload["quantiles"]),
            feature_columns=tuple(str(x) for x in payload["feature_columns"]),
            intercepts=tuple(float(x) for x in payload["intercepts"]),
            coefficients=tuple(tuple(float(v) for v in row) for row in payload["coefficients"]),
        )

    def predict(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        x = feature_frame.loc[:, list(self.feature_columns)].fillna(0.0).to_numpy(dtype=float)
        preds: dict[str, np.ndarray] = {}
        for quantile, intercept, coefs in zip(self.quantiles, self.intercepts, self.coefficients, strict=True):
            preds[f"{quantile:.4f}"] = intercept + x @ np.asarray(coefs, dtype=float)
        return pd.DataFrame(preds, index=feature_frame.index)


def build_probabilistic_feature_frame(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    cols = feature_columns or DEFAULT_PROB_FEATURE_COLUMNS
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0
    return out


def _infer_sigma_from_interval(
    lower: pd.Series,
    upper: pd.Series,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> pd.Series:
    z_span = float(norm.ppf(upper_quantile) - norm.ppf(lower_quantile))
    if abs(z_span) <= 1e-9:
        return pd.Series(0.0, index=lower.index, dtype=float)
    return ((upper - lower).abs() / z_span).astype(float)


class ProbabilisticForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        move_window: int = 100,
        vol_window: int = 100,
        model_path: str | Path | None = None,
        feature_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.config = config or ForecastConfig()
        self.move_window = move_window
        self.vol_window = vol_window
        self.model_path = Path(model_path) if model_path is not None else None
        self.feature_columns = feature_columns
        self.model = QuantileLinearModelArtifact.load(self.model_path) if self.model_path is not None else None

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        base = estimate_distribution(
            df=df,
            config=self.config,
            move_window=self.move_window,
            vol_window=self.vol_window,
        )
        out = base.copy()

        if self.model is None:
            out["forecast_source"] = "distribution_fallback"
            return compute_state_matrix_bands(out)

        features = build_probabilistic_feature_frame(
            out,
            feature_columns=self.model.feature_columns,
        )
        raw_preds = self.model.predict(features)
        q_map = {float(key): raw_preds[key] for key in raw_preds.columns}
        lower_q = self.config.probabilistic_lower_quantile
        upper_q = self.config.probabilistic_upper_quantile
        if lower_q not in q_map or 0.5 not in q_map or upper_q not in q_map:
            raise ValueError(
                "Probabilistic model artifact must include lower, median, and upper quantiles "
                f"({lower_q}, 0.5, {upper_q})."
            )

        out["forecast_return_lower"] = q_map[lower_q]
        out["forecast_return_median"] = q_map[0.5]
        out["forecast_return_upper"] = q_map[upper_q]
        out["forecast_return"] = out["forecast_return_median"]
        out["forecast_uncertainty"] = out["forecast_return_upper"] - out["forecast_return_lower"]
        out["mu"] = out["forecast_return_median"]
        out["sigma"] = _infer_sigma_from_interval(
            out["forecast_return_lower"],
            out["forecast_return_upper"],
            lower_quantile=lower_q,
            upper_quantile=upper_q,
        ).clip(lower=self.config.sigma_floor)
        out["mean_price"] = out["close"] * (1.0 + out["forecast_return"])
        out["forecast_source"] = "quantile_linear_model"

        return compute_state_matrix_bands(out)
