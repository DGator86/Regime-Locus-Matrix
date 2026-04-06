"""OCC option symbol parsing (US equity options), including Massive ``O:`` ticker prefix."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ParsedOccSymbol:
    root: str
    expiry: pd.Timestamp
    option_type: str  # "call" | "put"
    strike: float


def parse_occ_option_symbol(option_symbol: str) -> ParsedOccSymbol:
    """Parse OCC compact symbol (e.g. ``AAPL240202P00185000`` or ``O:AAPL240202P00185000``)."""
    s = str(option_symbol).strip().upper().replace(" ", "")
    if s.startswith("O:"):
        s = s[2:]
    if len(s) < 15:
        raise ValueError(f"Option symbol too short: {option_symbol!r}")

    strike8 = s[-8:]
    if not strike8.isdigit():
        raise ValueError(f"Unrecognized option symbol format: {option_symbol!r}")

    cp = s[-9]
    if cp not in ("C", "P"):
        raise ValueError(f"Unrecognized option symbol format: {option_symbol!r}")

    yymmdd = s[-15:-9]
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        raise ValueError(f"Unrecognized option symbol format: {option_symbol!r}")

    root = s[:-15]
    if not root:
        raise ValueError(f"Unrecognized option symbol format: {option_symbol!r}")

    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    year = 2000 + yy if yy < 70 else 1900 + yy
    expiry = pd.Timestamp(year=year, month=mm, day=dd)
    opt_type = "call" if cp == "C" else "put"
    strike = int(strike8) / 1000.0
    return ParsedOccSymbol(root=root, expiry=expiry, option_type=opt_type, strike=strike)
