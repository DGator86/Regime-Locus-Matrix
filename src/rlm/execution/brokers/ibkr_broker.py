"""IBKR broker adapter for normalizing trade execution responses."""

from __future__ import annotations

from typing import Any

from rlm.execution.brokers.base import BrokerAdapter


class IBKRBrokerAdapter(BrokerAdapter):
    broker = "ibkr"

    def submit_trade_decision(self, symbol: str, decision: dict[str, Any], paper: bool) -> dict[str, Any]:
        try:
            if str(decision.get("roee_action", "hold")).lower() == "hold":
                return {
                    "success": True,
                    "order_id": None,
                    "broker": self.broker,
                    "message": "hold action: no order submitted",
                    "details": {"paper": paper},
                }

            from rlm.execution.ibkr_combo_orders import (
                IBKROptionLegSpec,
                place_options_combo_market_order,
                reverse_legs_for_close,
                roee_side_to_ib_action,
            )

            spec = decision.get("ibkr_combo_spec") or {}
            legs_payload = list(spec.get("legs") or [])
            if not legs_payload:
                raise ValueError("decision missing ibkr_combo_spec.legs")

            legs = []
            for raw_leg in legs_payload:
                legs.append(
                    (
                        IBKROptionLegSpec(
                            underlying=str(spec.get("underlying") or symbol).upper(),
                            expiry_yyyymmdd=str(raw_leg["expiry"]).replace("-", ""),
                            strike=float(raw_leg["strike"]),
                            right="C" if str(raw_leg["option_type"]).lower() == "call" else "P",
                        ),
                        roee_side_to_ib_action(str(raw_leg["side"])),
                    )
                )

            if str(decision.get("roee_action", "enter")).lower() in {"exit", "close"}:
                legs = reverse_legs_for_close(legs)

            order_id, trail = place_options_combo_market_order(
                legs,
                quantity=int(decision.get("quantity", 1)),
                transmit=not paper,
                acknowledge_live=not paper,
            )
            return {
                "success": True,
                "order_id": str(order_id),
                "broker": self.broker,
                "message": "order submitted",
                "details": {"paper": paper, "status_trail": trail},
            }
        except Exception as exc:
            return {
                "success": False,
                "order_id": None,
                "broker": self.broker,
                "message": str(exc),
                "details": {"paper": paper},
            }
