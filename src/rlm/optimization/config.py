from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class NightlyHyperparams:
    """Nightly fine-tuned parameters (overlay on top of weekly regime model)."""

    mtf_ltf_weight: float = 0.5
    mtf_regimes: bool = True
    hmm_confidence_threshold: float = 0.65
    high_vol_kelly_multiplier: float = 0.6
    transition_kelly_multiplier: float = 0.8
    calm_trend_kelly_multiplier: float = 1.2
    move_window: int = 100
    vol_window: int = 100
    direction_neutral_threshold: float = 0.3
    transaction_cost_bps: float = 0.001

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "NightlyHyperparams":
        if not path:
            return cls()

        p = Path(path)
        if not p.exists():
            return cls()

        data = json.loads(p.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
