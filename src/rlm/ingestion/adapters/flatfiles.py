"""Massive flatfile ingest adapter (script logic moved from scripts/)."""

from __future__ import annotations

import re
from datetime import date, timedelta


def parse_date(raw: str) -> date:
    return date.fromisoformat(raw.strip())


def date_range(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def normalize_dataset_arg(raw: str) -> str:
    x = raw.strip().lower().replace("-", "_")
    aliases = {
        "trades": "trades",
        "quotes": "quotes",
        "day_aggs": "day_aggs",
        "dayaggs": "day_aggs",
        "minute_aggs": "minute_aggs",
        "minuteaggs": "minute_aggs",
    }
    if x not in aliases:
        raise ValueError(f"Unknown dataset {raw!r}; use trades, quotes, day-aggs, minute-aggs")
    return aliases[x]


def underlying_for_lake_path(*, underlying: str | None, allow_full_file: bool, prefixes: list[str]) -> str:
    if allow_full_file and not prefixes:
        return "BULK"
    if underlying:
        return str(underlying).strip().upper()
    if len(prefixes) == 1:
        m = re.match(r"^O:([A-Za-z]+)", str(prefixes[0]).strip().upper())
        if m:
            return m.group(1).upper()
    raise ValueError(
        "Pass --underlying SPY (or a single --ticker-prefix like O:SPY) so output paths "
        "use data/options/{UNDERLYING}/flatfiles/..."
    )
