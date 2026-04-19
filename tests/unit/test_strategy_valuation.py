from __future__ import annotations

import numpy as np

from rlm.training.strategy_structures import build_bull_call_structure, build_iron_condor_structure
from rlm.training.strategy_valuation import value_bull_call_at_expiry, value_iron_condor_path


def test_bull_call_value_caps_at_width() -> None:
    s = build_bull_call_structure(100.0, 5.0)
    assert value_bull_call_at_expiry(s, 120.0) == 5.0


def test_condor_path_penalizes_breaches() -> None:
    s = build_iron_condor_structure(100.0, 5.0)
    calm = value_iron_condor_path(s, np.array([100.0, 100.5, 99.8]))
    breach = value_iron_condor_path(s, np.array([100.0, 108.0, 110.0]))
    assert calm > breach
