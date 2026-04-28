from __future__ import annotations

import numpy as np

from rlm.features.scoring.regime_smoother import regime_flip_rate, smooth_regime_probabilities


def test_smoothing_reduces_flip_noise() -> None:
    raw = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.75, 0.25],
            [0.2, 0.8],
        ]
    )
    smooth = smooth_regime_probabilities(raw, alpha=0.2)
    raw_labels = raw.argmax(axis=1).tolist()
    smooth_labels = smooth.argmax(axis=1).tolist()
    assert regime_flip_rate([str(x) for x in smooth_labels]) <= regime_flip_rate([str(x) for x in raw_labels])
