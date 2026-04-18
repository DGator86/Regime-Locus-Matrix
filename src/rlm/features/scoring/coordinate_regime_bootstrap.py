from __future__ import annotations

from typing import Mapping

import numpy as np

from rlm.features.scoring.regime_model import REGIME_LABELS, RegimeModel

_BOOTSTRAP_MODEL = RegimeModel.with_bootstrap_coefficients()


def bootstrap_regime_label_from_coordinates(row: Mapping[str, float]) -> str:
    """Generate phase-1 pseudo labels via frozen bootstrap coefficients."""
    state = np.array(
        [
            [
                _safe_float(row, "M_D"),
                _safe_float(row, "M_V"),
                _safe_float(row, "M_L"),
                _safe_float(row, "M_G"),
                _safe_float(row, "M_trend_strength"),
                _safe_float(row, "M_dealer_control"),
                _safe_float(row, "M_alignment"),
                _safe_float(row, "M_delta_neutral"),
                _safe_float(row, "M_R_trans"),
            ]
        ],
        dtype=float,
    )
    probs = _BOOTSTRAP_MODEL.predict_proba(state)[0]
    return str(REGIME_LABELS[int(np.argmax(probs))])


def _safe_float(row: Mapping[str, float], key: str) -> float:
    raw = row.get(key, 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0
