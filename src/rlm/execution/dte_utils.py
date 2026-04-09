"""Days-to-expiry utilities for live position management.

``days_to_expiry`` returns fractional calendar days remaining, accounting for
the current time of day.  A value of 0.0 means the option expires today at
market close (or has already expired).  Negative values indicate expiry has
passed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Union

import pandas as pd

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:
    _ZoneInfo = None  # type: ignore[assignment,misc]

try:
    from dateutil.tz import gettz as _dateutil_gettz  # type: ignore[import-untyped]
except ImportError:
    _dateutil_gettz = None  # type: ignore[assignment]

# DST-aware US/Eastern timezone used to convert market-close times to UTC.
# Choosing the correct UTC offset (EDT = UTC-4, EST = UTC-5) matters: an
# April expiry at 16:00 ET is 20:00 UTC, not 21:00 UTC.
def _build_eastern() -> timezone | object:
    if _ZoneInfo is not None:
        return _ZoneInfo("America/New_York")
    if _dateutil_gettz is not None:
        tz = _dateutil_gettz("America/New_York")
        if tz is not None:
            return tz
    return timezone(timedelta(hours=-5))  # EST fallback (conservative)


_EASTERN = _build_eastern()

_US_MARKET_CLOSE_HOUR = 16  # 16:00 US/Eastern (UTC-4 EDT or UTC-5 EST depending on date)


def days_to_expiry(
    expiry: Union[str, "pd.Timestamp", datetime, None],
    *,
    reference_utc: datetime | None = None,
) -> float:
    """Return fractional calendar days until ``expiry`` close (16:00 ET assumed).

    Parameters
    ----------
    expiry:
        An ISO date string (``"2026-04-07"``), a ``pd.Timestamp``, or a
        ``datetime``.  Date-only values are treated as expiring at 16:00 ET
        on that calendar date.
    reference_utc:
        UTC datetime to use as "now" (default: ``datetime.now(timezone.utc)``).
        Override in tests.

    Returns
    -------
    float
        Fractional days remaining.  0.05 ≈ 72 minutes; < 0 means already past.
    """
    if expiry is None:
        return float("nan")

    now_utc = reference_utc if reference_utc is not None else datetime.now(timezone.utc)

    # Normalise expiry to a timezone-aware UTC datetime at 16:00 ET
    # (20:00 UTC during EDT, 21:00 UTC during EST).
    if isinstance(expiry, str):
        expiry = pd.Timestamp(expiry.strip())
    if isinstance(expiry, pd.Timestamp):
        expiry = expiry.to_pydatetime()

    if isinstance(expiry, datetime):
        if expiry.tzinfo is None:
            if expiry.hour != 0 or expiry.minute != 0:
                # Non-midnight time component already present — treat as UTC.
                expiry_utc = expiry.replace(tzinfo=timezone.utc)
            else:
                # Date-only / midnight: pin to US Eastern market close (16:00 ET)
                # and convert to UTC respecting DST (EDT = UTC-4, EST = UTC-5).
                expiry_utc = (
                    expiry.replace(
                        hour=_US_MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0,
                        tzinfo=_EASTERN,
                    ).astimezone(timezone.utc)
                )
        else:
            expiry_utc = expiry
    else:
        # fallback
        return float("nan")

    delta = expiry_utc - now_utc
    return delta.total_seconds() / 86_400.0


def dte_from_plan(plan: dict) -> float:
    """Extract the nearest-expiry DTE from a trade plan's ``matched_legs``."""
    mlegs = plan.get("matched_legs") or []
    if not mlegs:
        # Try ibkr_combo_spec legs
        spec = plan.get("ibkr_combo_spec") or {}
        mlegs = spec.get("legs") or []

    min_dte = float("inf")
    for leg in mlegs:
        exp = leg.get("expiry") or leg.get("expiration") or leg.get("expiry_date")
        if exp is None:
            continue
        d = days_to_expiry(exp)
        if not (d != d):  # not NaN
            min_dte = min(min_dte, d)

    return min_dte if min_dte != float("inf") else float("nan")


def needs_force_close(plan: dict, threshold_days: float) -> bool:
    """Return True when a plan's shortest-dated leg is within ``threshold_days`` of expiry.

    A ``threshold_days`` of 0.0 disables the check (always returns False).
    Typical values:
    - ``0.1``  ≈ ~2.4 hours  (recommended for 0DTE positions)
    - ``0.05`` ≈ ~72 minutes (tight)
    """
    if threshold_days <= 0.0:
        return False
    dte = dte_from_plan(plan)
    if dte != dte:  # NaN
        return False
    return dte <= threshold_days
