"""Broker adapter contracts for trade execution."""

from __future__ import annotations

from typing import Any


class BrokerAdapter:
    def submit_trade_decision(self, symbol: str, decision: dict[str, Any], paper: bool) -> dict[str, Any]:
        raise NotImplementedError
