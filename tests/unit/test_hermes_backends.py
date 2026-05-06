from __future__ import annotations

import pytest

from rlm.hermes_crew import backends as hermes_backends
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


def test_local_ollama_auto_selects_installed_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RLM_HERMES_BASE_URL", raising=False)
    monkeypatch.delenv("RLM_HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(hermes_backends, "_detect_ollama_model", lambda _base: "qwen2.5:7b-instruct")
    [(_, _, model)] = resolve_hermes_backend_tuples()
    assert model == "qwen2.5:7b-instruct"


def test_env_model_overrides_auto_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RLM_HERMES_BASE_URL", raising=False)
    monkeypatch.setenv("RLM_HERMES_MODEL", "my-fixed-model")
    monkeypatch.setattr(hermes_backends, "_detect_ollama_model", lambda _base: "qwen2.5:7b-instruct")
    [(_, _, model)] = resolve_hermes_backend_tuples()
    assert model == "my-fixed-model"


def test_qwen35_context_field_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_HERMES_OLLAMA_TAGS_TIMEOUT_SEC", raising=False)

    class _Resp:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def read(self) -> bytes:
            import json

            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def _fake_urlopen(req: object, timeout: float = 2.0) -> _Resp:
        _ = timeout
        full_url = getattr(req, "full_url", str(req))
        if str(full_url).endswith("/api/show"):
            return _Resp({"model_info": {"qwen35.context_length": 262144}})
        return _Resp({"models": [{"name": "qwen3.6:27b"}]})

    monkeypatch.setattr(hermes_backends, "urlopen", _fake_urlopen)
    model = hermes_backends._detect_ollama_model("http://127.0.0.1:11434/v1")
    assert model == "qwen3.6:27b"
