"""PDT challenge underlyings — SPY and QQQ only (single source of truth)."""

from __future__ import annotations

import os
import sys
from typing import Iterable

CHALLENGE_ALLOWED_UNDERLYINGS: tuple[str, ...] = ("SPY", "QQQ")
_CHALLENGE_ALLOWED_SET: frozenset[str] = frozenset(s.upper() for s in CHALLENGE_ALLOWED_UNDERLYINGS)


def normalize_challenge_underlying(symbol: str) -> str | None:
    """Return canonical upper symbol if allowed for the PDT challenge, else ``None``."""
    s = symbol.strip().upper()
    return s if s in _CHALLENGE_ALLOWED_SET else None


def _dedupe_preserve_order(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return tuple(out)


def resolve_challenge_underlyings_from_environ() -> tuple[str, ...]:
    """Resolve tickers for ``rlm challenge --run`` from the environment.

    Precedence:

    1. ``RLM_CHALLENGE_SYMBOLS`` — comma- or whitespace-separated list; unknown
       symbols are dropped with a warning.
    2. ``RLM_CHALLENGE_SYMBOL`` — a single ticker if allowed; invalid values log
       a warning and fall through to (3).
    3. Default: ``CHALLENGE_ALLOWED_UNDERLYINGS`` (SPY then QQQ).
    """
    raw_syms = (os.environ.get("RLM_CHALLENGE_SYMBOLS") or "").strip()
    raw_one = (os.environ.get("RLM_CHALLENGE_SYMBOL") or "").strip()

    if raw_syms:
        tokens = [t for t in raw_syms.replace(",", " ").split() if t.strip()]
        kept: list[str] = []
        dropped: list[str] = []
        for t in tokens:
            c = normalize_challenge_underlying(t)
            if c:
                kept.append(c)
            else:
                dropped.append(t.strip())
        if dropped:
            print(
                f"[warn] RLM_CHALLENGE_SYMBOLS ignored unknown ticker(s): {', '.join(dropped)} "
                f"(PDT challenge allows {', '.join(CHALLENGE_ALLOWED_UNDERLYINGS)} only)",
                file=sys.stderr,
                flush=True,
            )
        out = _dedupe_preserve_order(kept)
        if not out:
            print(
                f"[warn] RLM_CHALLENGE_SYMBOLS had no allowed tickers; "
                f"using {', '.join(CHALLENGE_ALLOWED_UNDERLYINGS)}",
                file=sys.stderr,
                flush=True,
            )
            return CHALLENGE_ALLOWED_UNDERLYINGS
        return out

    if raw_one:
        one = normalize_challenge_underlying(raw_one)
        if one:
            return (one,)
        print(
            f"[warn] RLM_CHALLENGE_SYMBOL={raw_one!r} is not allowed for the PDT challenge "
            f"({', '.join(CHALLENGE_ALLOWED_UNDERLYINGS)} only); using both defaults",
            file=sys.stderr,
            flush=True,
        )

    return CHALLENGE_ALLOWED_UNDERLYINGS
