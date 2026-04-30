from __future__ import annotations

from datetime import datetime, timezone

from app.adapters import RLMAdapter
from app.schemas import (
    DecisionSummary,
    FieldStatus,
    GammaVector,
    IVPoint,
    IVSurface,
    LiquidityWell,
    MarketFieldSnapshot,
    ParticleState,
    PricePathPoint,
    RegimeState,
    RegimeZone,
    SRWall,
    Vec3,
)


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
        return DecisionSummary(
            headline="Constructive, but confirmation is not clean yet.",
            details=details,
            risk_warning="Avoid chasing unless force alignment improves above 0.70.",
        )

    def generate_snapshot(self, symbol: str) -> MarketFieldSnapshot:
        rlm = self.adapter.get_snapshot_inputs(symbol)
        anchor_price = rlm["anchor_price"]
        current = rlm["current_price"]
        regime = RegimeState(
            label=rlm["regime_label"],
            bull_probability=rlm["regime_probabilities"]["bull"],
            bear_probability=rlm["regime_probabilities"]["bear"],
            chop_probability=rlm["regime_probabilities"]["chop"],
            confidence=rlm["confidence"],
        )
        field_status = FieldStatus(
            force_alignment=rlm["confidence"],
            volatility_pressure=0.42,
            liquidity_pull=0.74,
            gamma_bias="supportive",
            risk_state=rlm["risk_state"],
        )
        vel = self.compute_particle_velocity(rlm)
        decision = self.build_decision_summary(rlm, field_status)
        return MarketFieldSnapshot(
            symbol=rlm["symbol"],
            timestamp=datetime.now(timezone.utc),
            anchor_price=anchor_price,
            current_price=current,
            price_change=current - anchor_price,
            price_change_pct=((current - anchor_price) / anchor_price) * 100,
            regime=regime,
            field_status=field_status,
            particle=ParticleState(x=0.0, y=0.0, z=0.38, velocity=vel),
            regime_zones=self.build_regime_zones(rlm),
            gamma_vectors=[GammaVector(id="gex_1", origin=Vec3(x=-20, y=5, z=4), direction=Vec3(x=1, y=0.2, z=0.1), magnitude=0.62)],
            iv_surface=IVSurface(
                grid_size_x=32,
                grid_size_y=32,
                points=[IVPoint(x=-50, y=-20, z=4.2, iv=0.28), IVPoint(x=0, y=0, z=9.0, iv=0.41), IVPoint(x=50, y=20, z=7.5, iv=0.35)],
            ),
            liquidity_wells=[LiquidityWell(id="liq_5050", label="Liquidity Well", price=5050, x=-12, y=18, z=0, strength=0.81, type="supportive")],
            sr_walls=[
                SRWall(id="support_4950", label="Support Wall", price=4950, x=-24, height=22, strength=0.69, type="support"),
                SRWall(id="resistance_5210", label="Major Resistance", price=5210, x=24, height=35, strength=0.77, type="resistance"),
            ],
            price_path=[
                PricePathPoint(x=-30, y=-10, z=7, price=4980),
                PricePathPoint(x=-20, y=-5, z=8, price=5010),
                PricePathPoint(x=0, y=0, z=9, price=current),
            ],
            decision_summary=decision,
            recommended_action_label=self.ACTION_MAP.get(rlm["recommended_action"], "No clean trade"),
        )
