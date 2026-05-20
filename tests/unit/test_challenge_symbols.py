"""Tests for PDT challenge allowed underlyings (SPY / QQQ only)."""

from __future__ import annotations

import pytest

from rlm.challenge.symbols import (
    CHALLENGE_ALLOWED_UNDERLYINGS,
    normalize_challenge_underlying,
    resolve_challenge_underlyings_from_environ,
)


def test_normalize_accepts_spy_qqq() -> None:
    assert normalize_challenge_underlying("spy") == "SPY"
    assert normalize_challenge_underlying(" QQQ ") == "QQQ"


def test_normalize_rejects_other_tickers() -> None:
    assert normalize_challenge_underlying("AAPL") is None
    assert normalize_challenge_underlying("") is None


def test_resolve_default_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_CHALLENGE_SYMBOLS", raising=False)
    monkeypatch.delenv("RLM_CHALLENGE_SYMBOL", raising=False)
    assert resolve_challenge_underlyings_from_environ() == CHALLENGE_ALLOWED_UNDERLYINGS


def test_resolve_symbols_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_CHALLENGE_SYMBOLS", "QQQ, SPY")
    monkeypatch.delenv("RLM_CHALLENGE_SYMBOL", raising=False)
    assert resolve_challenge_underlyings_from_environ() == ("QQQ", "SPY")


def test_resolve_invalid_single_warns_and_defaults(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("RLM_CHALLENGE_SYMBOLS", raising=False)
    monkeypatch.setenv("RLM_CHALLENGE_SYMBOL", "IWM")
    out = resolve_challenge_underlyings_from_environ()
    err = capsys.readouterr().err
    assert out == CHALLENGE_ALLOWED_UNDERLYINGS
    assert "IWM" in err


def test_resolve_symbols_all_invalid_warns(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("RLM_CHALLENGE_SYMBOLS", "IWM, MSFT")
    monkeypatch.delenv("RLM_CHALLENGE_SYMBOL", raising=False)
    out = resolve_challenge_underlyings_from_environ()
    err = capsys.readouterr().err
    assert out == CHALLENGE_ALLOWED_UNDERLYINGS
    assert "ignored" in err.lower() or "no allowed" in err.lower()


def test_resolve_single_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_CHALLENGE_SYMBOLS", raising=False)
    monkeypatch.setenv("RLM_CHALLENGE_SYMBOL", "qqq")
    assert resolve_challenge_underlyings_from_environ() == ("QQQ",)
