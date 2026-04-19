from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VolumeProfileConfig:
    enabled: bool = False
    intraday_enabled: bool = False
    wyckoff_enabled: bool = False
    confluence_enabled: bool = False
    session_type: str = "equity"
    gating_enabled: bool = False
    wyckoff_threshold: float = 0.7
    balance_haircut: float = 0.5
    eighty_percent_boost: float = 0.2
    hybrid_strength_scaling: bool = True


@dataclass(frozen=True)
class FullRLMSharedConfig:
    volume_profile: VolumeProfileConfig = field(default_factory=VolumeProfileConfig)
