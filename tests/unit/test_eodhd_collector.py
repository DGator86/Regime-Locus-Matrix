"""EODHD collector scheduling."""

from __future__ import annotations

from rlm.data.eodhd_collector import default_poll_symbols


def test_default_poll_symbols_spy_qqq_first() -> None:
    syms = default_poll_symbols()
    assert syms[0] == "SPY"
    assert syms[1] == "QQQ"
    assert len(syms) == 12
