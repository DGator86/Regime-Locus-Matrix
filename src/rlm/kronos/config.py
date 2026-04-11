"""Pydantic configuration model for the Kronos integration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


class KronosConfig(BaseModel):
    """All tunables for the Kronos regime-confidence / forecast module."""

    model_name: str = "NeoQuasar/Kronos-mini"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k"
    max_context: int = 2048
    pred_len: int = Field(default=5, ge=1, description="Bars to predict ahead for regime assessment")
    temperature: float = Field(default=0.8, gt=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    sample_count: int = Field(default=10, ge=1, description="Independent sample paths per prediction")
    device: str = "cpu"
    finetuned_model_path: str | None = None
    cache_ttl_bars: int = Field(default=1, ge=0)
    regime_confidence_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    hmm_confidence_weight: float = Field(default=0.6, ge=0.0, le=1.0)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> KronosConfig:
        """Load from the ``kronos:`` block in a YAML config file."""
        cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            return cls()
        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh) or {}
        kronos_block = raw.get("kronos", {})
        if not kronos_block:
            return cls()
        return cls(**kronos_block)
