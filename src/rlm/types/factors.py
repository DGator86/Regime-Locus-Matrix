from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class FactorCategory(str, Enum):
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    DEALER_FLOW = "dealer_flow"


class TransformKind(str, Enum):
    RATIO = "ratio"
    SIGNED = "signed"


@dataclass(frozen=True)
class FactorSpec:
    name: str
    category: FactorCategory
    transform_kind: TransformKind
    neutral_value: float | None = None
    scale_value: float | None = None
    k: float = 1.0
    invert: bool = False


@dataclass
class FactorBundle:
    raw: Dict[str, float]
    standardized: Dict[str, float]
    composite_scores: Dict[str, float]
