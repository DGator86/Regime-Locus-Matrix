"""Post-hoc calibration for one-step-ahead regime top-1 probabilities.

Fits a monotone (isotonic) map from raw ``hmm_most_likely_next_prob`` (or Markov
equivalent) to empirical hit rates using historical forecast CSVs.  The fitted
curve is saved as JSON and applied at inference so reported top-1 probabilities
match realized frequency out-of-sample (on the training sample used for fitting).

Typical workflow::

    python scripts/fit_regime_transition_calibration.py --universe
    # writes data/processed/regime_transition_calibration.json

At runtime, if that file exists (or ``RLM_TRANSITION_CALIBRATION`` points to one),
pipelines add ``*_most_likely_next_prob_calibrated`` columns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

CalibKind = Literal["isotonic_top1_next_regime"]


def _interp_calibrated(p: np.ndarray, x_th: np.ndarray, y_th: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if x_th.size == 0:
        return p
    return np.interp(
        np.clip(p, float(x_th[0]), float(x_th[-1])),
        x_th,
        y_th,
    )


@dataclass(frozen=True)
class TransitionProbabilityCalibration:
    """Loaded calibration artifact (isotonic top-1 hit rate)."""

    kind: CalibKind
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray
    regime_family: str  # "hmm" | "markov"
    meta: dict[str, Any]

    def transform(self, p_hat: np.ndarray) -> np.ndarray:
        out = _interp_calibrated(
            np.asarray(p_hat, dtype=float),
            self.x_thresholds,
            self.y_thresholds,
        )
        return np.clip(out, 0.0, 1.0)


def default_calibration_path(data_root: str | Path | None) -> Path:
    from rlm.data.paths import get_processed_data_dir

    env = os.environ.get("RLM_TRANSITION_CALIBRATION", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return get_processed_data_dir(data_root) / "regime_transition_calibration.json"


def load_transition_calibration(
    data_root: str | Path | None = None,
    path: Path | None = None,
) -> TransitionProbabilityCalibration | None:
    """Load calibration JSON if present; return ``None`` if missing or invalid."""
    p = path or default_calibration_path(data_root)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if raw.get("kind") != "isotonic_top1_next_regime":
        return None
    try:
        x_th = np.asarray(raw["x_thresholds"], dtype=float)
        y_th = np.asarray(raw["y_thresholds"], dtype=float)
        fam = str(raw.get("regime_family", "hmm"))
    except (KeyError, ValueError, TypeError):
        return None
    if x_th.size < 2 or x_th.shape != y_th.shape:
        return None
    meta = {k: v for k, v in raw.items() if k not in ("x_thresholds", "y_thresholds", "kind")}
    return TransitionProbabilityCalibration(
        kind="isotonic_top1_next_regime",
        x_thresholds=x_th,
        y_thresholds=y_th,
        regime_family=fam,
        meta=meta,
    )


def apply_top1_calibration_inplace(
    df: Any,
    raw_col: str,
    out_col: str,
    cal: TransitionProbabilityCalibration | None,
) -> None:
    """Write calibrated probabilities column if *cal* is not ``None`` (in-place)."""
    if cal is None or raw_col not in df.columns:
        return
    raw = np.asarray(df[raw_col].to_numpy(dtype=float), dtype=float)
    df[out_col] = cal.transform(raw)


def fit_isotonic_top1_next_regime(
    p_hat: np.ndarray,
    pred_next_state: np.ndarray,
    actual_next_state: np.ndarray,
    *,
    regime_family: str = "hmm",
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Fit isotonic regression: *p_hat* → E[ 1{{pred_next == actual_next}} ].

    Arrays must be aligned (same length); typically use t..T-1 rows where
    *actual_next_state* is ``state[t+1]`` and *pred_next_state* is the model's
    argmax next state at *t*.
    """
    p_hat = np.asarray(p_hat, dtype=float).ravel()
    pred_next_state = np.asarray(pred_next_state, dtype=int).ravel()
    actual_next_state = np.asarray(actual_next_state, dtype=int).ravel()
    mask = (
        np.isfinite(p_hat)
        & (p_hat >= 0.0)
        & (p_hat <= 1.0)
        & np.isfinite(pred_next_state.astype(float))
        & np.isfinite(actual_next_state.astype(float))
    )
    p_hat = p_hat[mask]
    pred_next_state = pred_next_state[mask]
    actual_next_state = actual_next_state[mask]
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float).ravel()[mask]
    else:
        w = np.ones(len(p_hat))
    if len(p_hat) < 30:
        return None
    y = (pred_next_state == actual_next_state).astype(float)

    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return None

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_hat, y, sample_weight=w)
    x_th = np.asarray(iso.X_thresholds_, dtype=float)
    y_th = np.asarray(iso.y_thresholds_, dtype=float)
    if x_th.size < 2:
        return None
    return {
        "kind": "isotonic_top1_next_regime",
        "regime_family": regime_family,
        "x_thresholds": x_th.tolist(),
        "y_thresholds": y_th.tolist(),
        "n_samples": int(len(p_hat)),
        "base_rate": float(np.mean(y)),
        "mean_p_hat": float(np.mean(p_hat)),
    }


def save_calibration(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
