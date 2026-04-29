"""Kronos configuration — isolated from ``kronos_forecast`` to avoid import cycles with factor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KronosConfig:
    """Configuration for the Kronos foundation-model forecast layer."""

    model_name: str = "NeoQuasar/Kronos-mini"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k"
    device: str | None = "cpu"
    max_context: int = 2048
    clip: float = 5.0
    lookback: int = 200
    pred_len: int = 5
    sample_count: int = 10
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90
    sigma_floor: float = 1e-4
    stride: int = 1
    verbose: bool = False
    regime_confidence_weight: float = 0.4
    hmm_confidence_weight: float = 0.6
    finetuned_model_path: str | None = None
    cache_ttl_bars: int = 1

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> KronosConfig:
        """Load ``kronos:`` block from ``configs/default.yaml`` (repo root)."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyYAML is required for KronosConfig.from_yaml") from exc

        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = path or (repo_root / "configs" / "default.yaml")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        block = raw.get("kronos") or {}
        if not isinstance(block, dict):
            return cls()
        kwargs: dict[str, Any] = {}
        for field in (
            "model_name",
            "tokenizer_name",
            "device",
            "max_context",
            "clip",
            "lookback",
            "pred_len",
            "sample_count",
            "temperature",
            "top_p",
            "top_k",
            "lower_quantile",
            "upper_quantile",
            "sigma_floor",
            "stride",
            "verbose",
            "regime_confidence_weight",
            "hmm_confidence_weight",
            "finetuned_model_path",
            "cache_ttl_bars",
        ):
            if field in block and block[field] is not None:
                kwargs[field] = block[field]
        return cls(**kwargs)
