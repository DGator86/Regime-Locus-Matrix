from __future__ import annotations

import pytest

from rlm.trading_agents.config import _parse_int_env
from rlm.trading_agents.integration import _safe_float


# ── _safe_float ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (123.45, 123.45),
    ("123.45", 123.45),
    ("$123.45", 123.45),
    ("$1,234.56", 1234.56),
    (0, 0.0),
    (None, None),
    ("N/A", None),
    ("", None),
    ("n/a", None),
    ("--", None),
])
def test_safe_float(value, expected):
    assert _safe_float(value) == expected


# ── _parse_int_env ────────────────────────────────────────────────────────────

def test_parse_int_env_missing_key(monkeypatch):
    monkeypatch.delenv("TRADING_AGENTS_MAX_DEBATE_ROUNDS", raising=False)
    assert _parse_int_env("TRADING_AGENTS_MAX_DEBATE_ROUNDS", 1) == 1


def test_parse_int_env_valid(monkeypatch):
    monkeypatch.setenv("TRADING_AGENTS_MAX_DEBATE_ROUNDS", "3")
    assert _parse_int_env("TRADING_AGENTS_MAX_DEBATE_ROUNDS", 1) == 3


def test_parse_int_env_empty_string(monkeypatch):
    monkeypatch.setenv("TRADING_AGENTS_MAX_DEBATE_ROUNDS", "")
    assert _parse_int_env("TRADING_AGENTS_MAX_DEBATE_ROUNDS", 1) == 1


def test_parse_int_env_non_integer(monkeypatch):
    monkeypatch.setenv("TRADING_AGENTS_MAX_DEBATE_ROUNDS", "one")
    assert _parse_int_env("TRADING_AGENTS_MAX_DEBATE_ROUNDS", 1) == 1
