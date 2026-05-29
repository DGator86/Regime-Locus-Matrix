"""Tests for challenge underlying price resolution."""

from __future__ import annotations

from unittest.mock import patch

from rlm.challenge.underlying import resolve_session_underlying
from rlm.market.live_quotes import EquityQuote


class TestResolveSessionUnderlying:
    def test_override_wins(self) -> None:
        px, src = resolve_session_underlying("SPY", 711.0, override=755.0)
        assert px == 755.0
        assert src == "override"

    def test_prefers_live_over_stale_pipeline(self) -> None:
        quote = EquityQuote("SPY", 756.0, "2026-05-29T12:00:00+00:00", "test")
        with patch("rlm.challenge.underlying.live_marks_enabled", return_value=True), patch(
            "rlm.challenge.underlying.fetch_equity_quote", return_value=quote
        ):
            px, src = resolve_session_underlying(
                "SPY",
                711.52,
                pipeline_bar_date="2026-04-29",
                session_date="2026-05-29",
            )
        assert px == 756.0
        assert src == "live"

    def test_pipeline_fallback_when_live_unavailable(self) -> None:
        with patch("rlm.challenge.underlying.live_marks_enabled", return_value=False):
            px, src = resolve_session_underlying("SPY", 711.52)
        assert px == 711.52
        assert src == "pipeline"

    def test_hard_fallback_when_nothing_available(self) -> None:
        with patch("rlm.challenge.underlying.live_marks_enabled", return_value=False):
            px, src = resolve_session_underlying("SPY", None)
        assert px == 500.0
        assert src == "fallback"
