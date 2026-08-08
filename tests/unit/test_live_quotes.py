"""Live quote helpers (mocked — no network in CI)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.live_marks import refresh_challenge_state
from rlm.challenge.state import ChallengePosition, ChallengeState
from rlm.market.live_quotes import EquityQuote, fetch_equity_quote


class TestLiveQuotes:
    def test_refresh_challenge_updates_premium(self) -> None:
        cfg = ChallengeConfig(symbol="SPY")
        pos = ChallengePosition.new(
            symbol="SPY",
            option_type="call",
            direction="long",
            underlying_entry=700.0,
            strike=740.0,
            dte=7,
            entry_date="2026-05-28",
            premium_per_share=3.0,
            qty=1,
            delta=0.45,
            iv=0.18,
        )
        state = ChallengeState(balance=700.0, seed=1_000.0, target=25_000.0, open_positions=[pos])

        quote = EquityQuote("SPY", 710.0, "2026-05-28T12:00:00+00:00", "test")
        with (
            patch("rlm.challenge.live_marks.live_marks_enabled", return_value=True),
            patch("rlm.challenge.live_marks.fetch_equity_quote", return_value=quote),
            patch("rlm.challenge.live_marks.fetch_option_mid_per_share", return_value=3.5),
        ):
            asof = refresh_challenge_state(state, cfg, session_date="2026-05-28")

        assert asof is not None
        assert pos.current_premium == 3.5
        assert pos.unrealised_pnl == 50.0

    def test_refresh_challenge_arms_trail_when_peak_crosses_activate(self) -> None:
        cfg = ChallengeConfig(symbol="SPY")
        pos = ChallengePosition.new(
            symbol="SPY",
            option_type="call",
            direction="long",
            underlying_entry=700.0,
            strike=700.0,
            dte=7,
            entry_date="2026-05-28",
            premium_per_share=2.0,
            qty=1,
            delta=0.50,
            iv=0.18,
        )
        assert pos.trail_armed is False
        state = ChallengeState(balance=700.0, seed=1_000.0, target=100_000.0, open_positions=[pos])
        # 1.30× > stage1 trail activate 1.22
        quote = EquityQuote("SPY", 710.0, "2026-05-28T12:00:00+00:00", "test")
        with (
            patch("rlm.challenge.live_marks.live_marks_enabled", return_value=True),
            patch("rlm.challenge.live_marks.fetch_equity_quote", return_value=quote),
            patch("rlm.challenge.live_marks.fetch_option_mid_per_share", return_value=2.60),
        ):
            refresh_challenge_state(state, cfg, session_date="2026-05-28")

        assert pos.peak_premium_mult == pytest.approx(1.30)
        assert pos.trail_armed is True

    def test_fetch_equity_quote_uses_fast_info(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.fast_info = {"lastPrice": 512.34}
        with patch("yfinance.Ticker", return_value=mock_ticker):
            q = fetch_equity_quote("SPY")
        assert q is not None
        assert q.price == 512.34
