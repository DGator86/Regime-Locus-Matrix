from __future__ import annotations

import numpy as np


def smooth_regime_probabilities(raw_probs: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    if raw_probs.ndim != 2:
        raise ValueError("raw_probs must have shape (T, K)")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if raw_probs.shape[0] == 0:
        return raw_probs.copy()
    out = np.zeros_like(raw_probs, dtype=float)
    out[0] = raw_probs[0]
    out[0] /= np.maximum(out[0].sum(), 1e-12)
    for t in range(1, raw_probs.shape[0]):
        out[t] = alpha * raw_probs[t] + (1.0 - alpha) * out[t - 1]
        out[t] /= np.maximum(out[t].sum(), 1e-12)
    return out


def regime_flip_rate(labels: list[str]) -> float:
    if len(labels) < 2:
        return 0.0
    flips = sum(labels[i] != labels[i - 1] for i in range(1, len(labels)))
    return flips / (len(labels) - 1)
