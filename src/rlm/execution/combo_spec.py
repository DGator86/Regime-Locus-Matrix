"""Vendor-neutral multi-leg option combo JSON on trade plans.

Structured specs use ``combo_spec`` (``underlying``, ``legs[]``, ``quantity``, …).
The legacy key ``ibkr_combo_spec`` is still accepted for older files. RLM does not
route option orders through Interactive Brokers — this module is for parsing,
monitoring, and local bookkeeping only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

ComboLegAction = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class OptionComboLeg:
    """Single option leg described in a stored combo JSON."""

    underlying: str
    expiry_yyyymmdd: str
    strike: float
    right: str  # "C" or "P"
    exchange: str = "SMART"
    currency: str = "USD"
    multiplier: str = "100"


def expiry_iso_to_yyyymmdd(expiry: str) -> str:
    s = str(expiry).strip().replace("-", "")
    if len(s) == 8 and s.isdigit():
        return s
    return pd.Timestamp(expiry).strftime("%Y%m%d")


def side_to_combo_action(side: str) -> ComboLegAction:
    s = str(side).lower().strip()
    if s == "long":
        return "BUY"
    if s == "short":
        return "SELL"
    raise ValueError(f"leg side must be 'long' or 'short', got {side!r}")


def option_type_to_right(option_type: str) -> str:
    t = str(option_type).lower().strip()
    if t == "call":
        return "C"
    if t == "put":
        return "P"
    raise ValueError(f"option_type must be call or put, got {option_type!r}")


def legs_from_combo_spec(spec: dict[str, Any]) -> list[tuple[OptionComboLeg, ComboLegAction]]:
    und = str(spec.get("underlying", "")).upper().strip()
    if not und:
        raise ValueError("combo_spec missing underlying")
    raw_legs = spec.get("legs")
    if not isinstance(raw_legs, list) or not raw_legs:
        raise ValueError("combo_spec.legs must be a non-empty list")
    out: list[tuple[OptionComboLeg, ComboLegAction]] = []
    for leg in raw_legs:
        if not isinstance(leg, dict):
            raise ValueError("each leg must be an object")
        sp = OptionComboLeg(
            underlying=und,
            expiry_yyyymmdd=expiry_iso_to_yyyymmdd(str(leg["expiry"])),
            strike=float(leg["strike"]),
            right=option_type_to_right(str(leg["option_type"])),
        )
        act = side_to_combo_action(str(leg["side"]))
        out.append((sp, act))
    return out


def reverse_combo_legs(
    legs: list[tuple[OptionComboLeg, ComboLegAction]],
) -> list[tuple[OptionComboLeg, ComboLegAction]]:
    rev: list[tuple[OptionComboLeg, ComboLegAction]] = []
    for spec, act in legs:
        new_act: ComboLegAction = "SELL" if act == "BUY" else "BUY"
        rev.append((spec, new_act))
    return rev


def plan_combo_spec(plan: dict[str, Any]) -> dict[str, Any] | None:
    s = plan.get("combo_spec")
    if isinstance(s, dict):
        return s
    s = plan.get("ibkr_combo_spec")
    if isinstance(s, dict):
        return s
    return None
