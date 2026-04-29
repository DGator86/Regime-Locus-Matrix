from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_pipeline_event(
    *,
    symbol: str,
    bar_id: str,
    factor_values: dict[str, float | None],
    regime_state: str,
    kronos_confidence: float | None,
    action: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "bar_id": bar_id,
        "factor_values": factor_values,
        "regime_state": regime_state,
        "kronos_confidence": kronos_confidence,
        "action": action,
    }
    if extra:
        payload.update(extra)
    return payload

