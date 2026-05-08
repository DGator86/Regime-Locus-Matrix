"""Tests for regime-based challenge directive fallback."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from rlm.challenge.regime_signal import regime_fallback_directive_from_policy_row


@pytest.fixture(autouse=True)
def _challenge_fallback_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_CHALLENGE_REGIME_FALLBACK", raising=False)


def test_direction_regime_bull(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_CHALLENGE_REGIME_FALLBACK", raising=False)
    row = pd.Series({"direction_regime": "bull", "regime_key": "range|high_vol|low_liq|neutral"})
    d, note = regime_fallback_directive_from_policy_row(row)
    assert d == "long"
    assert "direction_regime=bull" in note


def test_regime_key_bear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_CHALLENGE_REGIME_FALLBACK", raising=False)
    row = pd.Series({"direction_regime": "neutral", "regime_key": "bear|low_vol|low_liq|neutral"})
    d, note = regime_fallback_directive_from_policy_row(row)
    assert d == "short"
    assert "regime_key" in note.lower()


def test_range_regime_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_CHALLENGE_REGIME_FALLBACK", raising=False)
    row = pd.Series({"direction_regime": "neutral", "regime_key": "range|high_vol|low_liq|supportive"})
    d, note = regime_fallback_directive_from_policy_row(row)
    assert d is None
    assert "suppressed" in note


def test_disabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_CHALLENGE_REGIME_FALLBACK", "0")
    row = pd.Series({"direction_regime": "bull"})
    d, note = regime_fallback_directive_from_policy_row(row)
    assert d is None
    assert "disabled" in note
