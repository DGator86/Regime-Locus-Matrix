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

            spec = decision.get("combo_spec") or decision.get("ibkr_combo_spec") or {}
            legs_payload = list(spec.get("legs") or [])
            if not legs_payload:
                raise ValueError("decision missing combo_spec.legs (equities-only IBKR adapter)")

            return {
                "success": False,
                "order_id": None,
                "broker": self.broker,
                "message": "RLM policy: IBKR is equities-only; option combos are not submitted",
                "details": {"paper": paper},
            }
        except Exception as exc:
            return {
                "success": False,
                "order_id": None,
                "broker": self.broker,
                "message": str(exc),
                "details": {"paper": paper},
            }
