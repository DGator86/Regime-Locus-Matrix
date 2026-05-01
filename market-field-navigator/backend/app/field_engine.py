from __future__ import annotations

from datetime import datetime, timezone
import math

from app.adapters import RLMAdapter
from app.schemas import DecisionSummary, FieldStatus, GammaVector, IVPoint, IVSurface, LiquidityWell, MarketFieldSnapshot, ParticleState, PricePathPoint, RegimeState, RegimeZone, SRWall, Vec3


class MarketFieldEngine:
    ACTION_MAP = {
        "ENTER_LONG": "Long setup active",
        "ENTER_SHORT": "Short setup active",
        "WAIT": "No clean trade",
        "WAIT_FOR_CONFIRMATION": "Setup forming, confirmation needed",
        "HEDGE": "Defensive / hedge environment",
        "EXIT": "Exit / reduce exposure",
    }

    def __init__(self, adapter: RLMAdapter | None = None):
        self.adapter = adapter or RLMAdapter()

    @staticmethod
    def price_to_x(price: float, anchor: float) -> float:
        return ((price - anchor) / anchor) * 1000

    def build_regime_zones(self, rlm: dict) -> list[RegimeZone]:
        confidence = rlm["confidence"]
        probs = rlm["regime_probabilities"]
        return [
            RegimeZone(id="bull_zone", label="Bull Regime", type="bull", center=Vec3(x=-35, y=15, z=14), size=Vec3(x=40 + probs["bull"] * 30, y=30, z=20), opacity=0.15 + confidence * 0.25, probability=probs["bull"]),
            RegimeZone(id="bear_zone", label="Bear Regime", type="bear", center=Vec3(x=35, y=15, z=14), size=Vec3(x=40 + probs["bear"] * 30, y=30, z=20), opacity=0.10 + confidence * 0.20, probability=probs["bear"]),
            RegimeZone(id="chop_zone", label="Chop Zone", type="chop", center=Vec3(x=0, y=8, z=10), size=Vec3(x=25 + probs["chop"] * 30, y=20, z=14), opacity=0.08 + confidence * 0.12, probability=probs["chop"]),
        ]

    def compute_particle_velocity(self, rlm: dict) -> Vec3:
        t = rlm["transition_probabilities"]
        state = rlm["hmm_state"]
        if state == "trend_continuation":
            return Vec3(x=0.02 + t.get("trend_continuation", 0) * 0.03, y=0.02, z=0.01)
        if state == "mean_reversion":
            return Vec3(x=-0.02, y=0.01, z=0.0)
        if state == "breakout":
            return Vec3(x=0.05, y=0.03, z=0.02)
        if state == "vol_expansion":
            return Vec3(x=0.01, y=0.01, z=0.04)
        return Vec3(x=0.005, y=0.01, z=0.005)

    def build_decision_summary(self, rlm: dict, field_status: FieldStatus) -> DecisionSummary:
        details = [d["explanation"] for d in rlm["top_drivers"]]
        details.append(f"Force alignment is {field_status.force_alignment:.2f}, below breakout-confirmation threshold.")
        return DecisionSummary(headline="Constructive, but confirmation is not clean yet.", details=details, risk_warning="Avoid chasing unless force alignment improves above 0.70.")

    def build_iv_surface(self, grid_x: int = 32, grid_y: int = 32) -> IVSurface:
        points: list[IVPoint] = []
        for ix in range(grid_x):
            for iy in range(grid_y):
                x = -50 + (100 * ix / (grid_x - 1))
                y = -20 + (40 * iy / (grid_y - 1))
                wave = 0.25 * math.sin(ix / 4) + 0.2 * math.cos(iy / 5)
                iv = 0.24 + (ix / grid_x) * 0.12 + (iy / grid_y) * 0.05 + wave * 0.02
                points.append(IVPoint(x=x, y=y, z=iv * 30, iv=iv))
        return IVSurface(grid_size_x=grid_x, grid_size_y=grid_y, points=points)

    def generate_snapshot(self, symbol: str) -> MarketFieldSnapshot:
        rlm = self.adapter.get_snapshot_inputs(symbol)
        anchor_price = rlm["anchor_price"]
        current = rlm["current_price"]
        regime = RegimeState(label=rlm["regime_label"], bull_probability=rlm["regime_probabilities"]["bull"], bear_probability=rlm["regime_probabilities"]["bear"], chop_probability=rlm["regime_probabilities"]["chop"], confidence=rlm["confidence"])
        field_status = FieldStatus(force_alignment=max(0.0, min(1.0, rlm["confidence"])), volatility_pressure=0.42, liquidity_pull=0.74, gamma_bias="supportive", risk_state=rlm["risk_state"])
        vel = self.compute_particle_velocity(rlm)
        decision = self.build_decision_summary(rlm, field_status)
        supports = sorted(rlm["levels"]["support"], reverse=True)
        resistances = sorted(rlm["levels"]["resistance"])
        sr_walls = [
            *[SRWall(id=f"support_{int(p)}", label="Support Wall", price=p, x=self.price_to_x(p, anchor_price), height=20 + i * 4, strength=max(0.4, 0.75 - i * 0.08), type="support") for i, p in enumerate(supports)],
            *[SRWall(id=f"resistance_{int(p)}", label="Resistance Wall", price=p, x=self.price_to_x(p, anchor_price), height=24 + i * 4, strength=max(0.45, 0.82 - i * 0.08), type="resistance") for i, p in enumerate(resistances)],
        ]
        liquidity = [LiquidityWell(id=f"liq_{int(p)}", label="Liquidity Well", price=p, x=self.price_to_x(p, anchor_price), y=14 + idx * 4, z=0, strength=0.55 + 0.1 * (idx % 3), type="supportive" if p < current else "overhead") for idx, p in enumerate(supports + resistances)]
        gamma = [GammaVector(id=f"gex_{i}", origin=Vec3(x=-30 + i * 10, y=4 + (i % 3) * 3, z=3 + i * 0.4), direction=Vec3(x=1, y=0.1 * ((i % 2) * 2 - 1), z=0.05 * (i % 4)), magnitude=0.35 + i * 0.07) for i in range(6)]
        return MarketFieldSnapshot(
            symbol=rlm["symbol"], timestamp=datetime.now(timezone.utc), anchor_price=anchor_price, current_price=current, price_change=current - anchor_price, price_change_pct=((current - anchor_price) / anchor_price) * 100,
            regime=regime, field_status=field_status,
            particle=ParticleState(x=self.price_to_x(current, anchor_price), y=0.0, z=0.38, velocity=vel), regime_zones=self.build_regime_zones(rlm), gamma_vectors=gamma,
            iv_surface=self.build_iv_surface(), liquidity_wells=liquidity, sr_walls=sr_walls,
            price_path=[PricePathPoint(x=self.price_to_x(p, anchor_price), y=-10 + i * 5, z=7 + i, price=p) for i, p in enumerate([anchor_price - 65, anchor_price - 35, current])],
            decision_summary=decision, recommended_action_label=self.ACTION_MAP.get(rlm["recommended_action"], "No clean trade"),
        )
