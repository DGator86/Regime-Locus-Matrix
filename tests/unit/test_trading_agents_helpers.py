from __future__ import annotations

import pytest

from rlm.trading_agents.config import _GROQ_BASE_URL, _auto_detect_provider, _parse_int_env
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


# ── _auto_detect_provider ─────────────────────────────────────────────────────

_ALL_KEYS = (
    "GROQ_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


def _clear_all_keys(monkeypatch):
    for k in _ALL_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_auto_detect_groq_priority(monkeypatch):
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza_test")  # lower priority
    provider, deep, quick, backend = _auto_detect_provider()
    assert provider == "openai"
    assert backend == _GROQ_BASE_URL
    assert "llama" in deep.lower()


def test_auto_detect_google_when_no_groq(monkeypatch):
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza_test")
    provider, deep, quick, backend = _auto_detect_provider()
    assert provider == "google"
    assert "gemini" in deep.lower()
    assert backend is None


def test_auto_detect_openrouter_fallback(monkeypatch):
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    provider, deep, quick, backend = _auto_detect_provider()
    assert provider == "openrouter"
    assert backend is None


def test_auto_detect_anthropic_fallback(monkeypatch):
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    provider, deep, quick, backend = _auto_detect_provider()
    assert provider == "anthropic"


def test_auto_detect_openai_last_resort(monkeypatch):
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider, deep, quick, backend = _auto_detect_provider()
    assert provider == "openai"
    assert backend is None  # no Groq endpoint when using native OpenAI
