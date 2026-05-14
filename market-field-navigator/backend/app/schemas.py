from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class RegimeState(BaseModel):
    label: str
    bull_probability: float
    bear_probability: float
    chop_probability: float
    confidence: float


class FieldStatus(BaseModel):
    force_alignment: float
    volatility_pressure: float
    liquidity_pull: float
    gamma_bias: str
    risk_state: str


class ParticleState(BaseModel):
    x: float
    y: float
    z: float
    velocity: Vec3


class RegimeZone(BaseModel):
    id: str
    label: str
    type: Literal["bull", "bear", "chop"]
    center: Vec3
    size: Vec3
    opacity: float
    probability: float


class GammaVector(BaseModel):
    id: str
    origin: Vec3
    direction: Vec3
    magnitude: float


class IVPoint(BaseModel):
    x: float
    y: float
    z: float
    iv: float


class IVSurface(BaseModel):
    grid_size_x: int
    grid_size_y: int
    points: list[IVPoint]


class LiquidityWell(BaseModel):
    id: str
    label: str
    price: float
    x: float
    y: float
    z: float
    strength: float
    type: str


class SRWall(BaseModel):
    id: str
    label: str
    price: float
    x: float
    height: float
    strength: float
    type: Literal["support", "resistance"]


class PricePathPoint(BaseModel):
    x: float
    y: float
    z: float
    price: float


class DecisionSummary(BaseModel):
    headline: str
    details: list[str]
    risk_warning: str


class MarketFieldSnapshot(BaseModel):
    symbol: str
    timestamp: datetime
    anchor_price: float
    current_price: float
    price_change: float
    price_change_pct: float
    regime: RegimeState
    field_status: FieldStatus
    particle: ParticleState
    regime_zones: list[RegimeZone]
    gamma_vectors: list[GammaVector]
    iv_surface: IVSurface
    liquidity_wells: list[LiquidityWell]
    sr_walls: list[SRWall]
    price_path: list[PricePathPoint]
    decision_summary: DecisionSummary
    recommended_action_label: str
