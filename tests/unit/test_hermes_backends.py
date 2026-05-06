from __future__ import annotations

import os

import pytest

from rlm.hermes_crew.backends import resolve_hermes_backend_tuples


def test_resolve_prefers_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("RLM_HERMES_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("RLM_HERMES_API_KEY", "secret")
    monkeypatch.setenv("RLM_HERMES_MODEL", "m1")
    [(base, key, model)] = resolve_hermes_backend_tuples()
    assert base == "https://example.com/v1"
    assert key == "secret"
    assert model == "m1"


def test_groq_requires_auto_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.delenv("RLM_HERMES_BASE_URL", raising=False)
    monkeypatch.delenv("RLM_HERMES_AUTO_GROQ", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    [(base, key, model)] = resolve_hermes_backend_tuples()
    assert "11434" in base
    assert key == "ollama"


def test_groq_auto_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("RLM_HERMES_AUTO_GROQ", "1")
    monkeypatch.delenv("RLM_HERMES_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    [(base, key, _model)] = resolve_hermes_backend_tuples()
    assert base == "https://api.groq.com/openai/v1"
    assert key == "gsk_test"


def test_openrouter_when_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("RLM_HERMES_BASE_URL", raising=False)
    monkeypatch.delenv("RLM_HERMES_AUTO_GROQ", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    [(base, key, model)] = resolve_hermes_backend_tuples()
    assert base == "https://openrouter.ai/api/v1"
    assert key == "sk-or-test"
    assert "meta-llama" in model


def test_openrouter_precedence_over_groq_key_without_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("RLM_HERMES_BASE_URL", raising=False)
    monkeypatch.delenv("RLM_HERMES_AUTO_GROQ", raising=False)
    [(base, key, _)] = resolve_hermes_backend_tuples()
    assert "openrouter" in base
    assert key == "sk-or-test"
