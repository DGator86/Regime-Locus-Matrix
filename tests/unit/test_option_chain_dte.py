"""Calendar-day DTE for normalized option chains (Massive intraday timestamps)."""

from __future__ import annotations

import pandas as pd

from rlm.data.option_chain import normalize_option_chain, select_nearest_expiry_slice


def _raw_chain(*, timestamp: pd.Timestamp, expiries: list[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp in expiries:
        rows.append(
            {
                "timestamp": timestamp,
                "underlying": "SPY",
                "expiry": exp,
                "option_type": "call",
                "strike": 100.0,
                "bid": 1.0,
                "ask": 1.1,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_option_chain_dte_is_calendar_days_with_intraday_timestamp() -> None:
    """RTH Massive ``timestamp=now`` must not undercount DTE via Timedelta.days."""
    session = pd.Timestamp("2026-08-10 19:00:00")  # 15:00 ET during RTH
    expiries = [pd.Timestamp("2026-08-10") + pd.Timedelta(days=n) for n in (0, 1, 7, 21, 22)]
    chain = normalize_option_chain(_raw_chain(timestamp=session, expiries=expiries))
    by_exp = {pd.Timestamp(r.expiry).normalize(): int(r.dte) for r in chain.itertuples()}

    assert by_exp[pd.Timestamp("2026-08-10")] == 0  # true 0DTE (was -1)
    assert by_exp[pd.Timestamp("2026-08-11")] == 1
    assert by_exp[pd.Timestamp("2026-08-17")] == 7
    assert by_exp[pd.Timestamp("2026-08-31")] == 21
    assert by_exp[pd.Timestamp("2026-09-01")] == 22


def test_short_dte_window_includes_0dte_during_rth() -> None:
    session = pd.Timestamp("2026-08-10 19:00:00")
    expiries = [pd.Timestamp("2026-08-10") + pd.Timedelta(days=n) for n in range(0, 8)]
    chain = normalize_option_chain(_raw_chain(timestamp=session, expiries=expiries))
    eligible = chain[(chain["dte"] >= 0) & (chain["dte"] <= 5)]
    assert 0 in set(int(x) for x in eligible["dte"])
    assert pd.Timestamp("2026-08-10") in set(pd.Timestamp(e).normalize() for e in eligible["expiry"])
    # Same-day expiry is selectable when it is the only contract in-window.
    only_0dte = chain[chain["expiry"].dt.normalize() == pd.Timestamp("2026-08-10")]
    slice_ = select_nearest_expiry_slice(only_0dte, dte_min=0, dte_max=5)
    assert not slice_.empty
    assert int(slice_["dte"].iloc[0]) == 0


def test_three_track_window_keeps_calendar_7_and_excludes_22() -> None:
    session = pd.Timestamp("2026-08-10 19:00:00")
    expiries = [pd.Timestamp("2026-08-10") + pd.Timedelta(days=n) for n in range(0, 25)]
    chain = normalize_option_chain(_raw_chain(timestamp=session, expiries=expiries))
    eligible = chain[(chain["dte"] >= 7) & (chain["dte"] <= 21)]
    cal_offsets = sorted({(pd.Timestamp(e).normalize() - pd.Timestamp("2026-08-10")).days for e in eligible["expiry"]})
    assert 7 in cal_offsets
    assert 21 in cal_offsets
    assert 6 not in cal_offsets
    assert 22 not in cal_offsets
