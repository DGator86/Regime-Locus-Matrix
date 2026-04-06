"""Collect all pages from a Massive list response (``results`` + ``next_url``)."""

from __future__ import annotations

from typing import Any

from rlm.data.massive import MassiveClient


def collect_massive_results(client: MassiveClient, first: dict[str, Any] | None) -> list[Any]:
    if not first or not isinstance(first, dict):
        return []
    rows: list[Any] = list(first.get("results", []))
    next_url = first.get("next_url")
    while next_url:
        data = client.get_by_url(str(next_url))
        if not data or not isinstance(data, dict):
            break
        rows.extend(data.get("results", []))
        next_url = data.get("next_url")
    return rows
