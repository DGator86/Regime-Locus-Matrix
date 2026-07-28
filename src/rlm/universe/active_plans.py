"""
Active trade-plan selection from universe JSON.

Matches equity monitor intent: union of ``active_ranked`` and ``results``,
``status == "active"`` only, deduped by ``plan_id`` (first occurrence wins —
``active_ranked`` rows precede ``results`` in the walk order).
"""

from __future__ import annotations

from typing import Any


def iter_active_trade_plan_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Return active plan row dicts in stable order (ranked segment first)."""
    plans: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in (raw.get("active_ranked") or []) + (raw.get("results") or []):
        if not (isinstance(row, dict) and row.get("status") == "active"):
            continue
        pid = str(row.get("plan_id") or "")
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)
        plans.append(row)
    return plans


def active_plan_ids(raw: dict[str, Any]) -> set[str]:
    """Set of ``plan_id`` strings for active universe rows."""
    return {str(r.get("plan_id")) for r in iter_active_trade_plan_rows(raw) if r.get("plan_id")}


def active_symbols(raw: dict[str, Any]) -> set[str]:
    """Set of underlying symbols for active universe rows."""
    out: set[str] = set()
    for r in iter_active_trade_plan_rows(raw):
        sym = str(r.get("symbol") or "").strip().upper()
        if sym:
            out.add(sym)
    return out


def symbol_by_plan_id(raw: dict[str, Any]) -> dict[str, str]:
    """Map plan_id → underlying symbol for active rows."""
    m: dict[str, str] = {}
    for r in iter_active_trade_plan_rows(raw):
        pid = str(r.get("plan_id") or "")
        if pid:
            m[pid] = str(r.get("symbol") or "?")
    return m
