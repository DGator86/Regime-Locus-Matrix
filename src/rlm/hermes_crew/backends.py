"""Resolve OpenAI-compatible LLM endpoints for the Hermes crew."""

from __future__ import annotations

import os

_OPENROUTER_URL = "https://openrouter.ai/api/v1"
# OpenRouter free-tier models use the `:free` suffix (quota subject to their policy).
_DEFAULT_OPENROUTER_FREE_MODEL = "meta-llama/llama-3.2-3b-instruct:free"


def _truthy(key: str) -> bool:
    return (os.environ.get(key) or "").strip().lower() in ("1", "true", "yes", "on")


def resolve_hermes_backend_tuples() -> list[tuple[str, str, str]]:
    """
    Ordered list of (base_url, api_key, model) for primary and optional fallback backends.

    Precedence when ``RLM_HERMES_BASE_URL`` is unset:

    1. Groq — only if ``RLM_HERMES_AUTO_GROQ`` is truthy and ``GROQ_API_KEY`` is set.
    2. OpenRouter — if ``OPENROUTER_API_KEY`` is set (recommended free cloud default).
    3. Local Ollama — ``http://127.0.0.1:11434/v1``.

    ``GROQ_API_KEY`` alone does **not** enable Groq (avoids silent daily cap failures when
    you meant to use Ollama or OpenRouter).
    """
    explicit_base = (os.environ.get("RLM_HERMES_BASE_URL") or "").strip()
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()

    if explicit_base:
        key = (os.environ.get("RLM_HERMES_API_KEY") or "").strip() or "ollama"
        model = (os.environ.get("RLM_HERMES_MODEL") or os.environ.get("LLM_MODEL") or "").strip() or "llama3.2"
        primary: tuple[str, str, str] = (explicit_base, key, model)
    elif _truthy("RLM_HERMES_AUTO_GROQ") and groq_key:
        model = (
            (os.environ.get("RLM_HERMES_MODEL") or os.environ.get("LLM_MODEL") or "").strip()
            or "llama-3.3-70b-versatile"
        )
        primary = ("https://api.groq.com/openai/v1", groq_key, model)
    elif openrouter_key:
        model = (
            (os.environ.get("RLM_HERMES_MODEL") or os.environ.get("LLM_MODEL") or "").strip()
            or (os.environ.get("RLM_HERMES_OPENROUTER_MODEL") or "").strip()
            or _DEFAULT_OPENROUTER_FREE_MODEL
        )
        primary = (_OPENROUTER_URL, openrouter_key, model)
    else:
        model = (os.environ.get("RLM_HERMES_MODEL") or os.environ.get("LLM_MODEL") or "").strip() or "llama3.2"
        primary = ("http://127.0.0.1:11434/v1", "ollama", model)

    fallback_model = (os.environ.get("RLM_HERMES_FALLBACK_MODEL") or "").strip()
    fallback_base = (os.environ.get("RLM_HERMES_FALLBACK_BASE_URL") or "").strip()
    fallback_key = (os.environ.get("RLM_HERMES_FALLBACK_API_KEY") or "").strip()
    if fallback_model and not fallback_base:
        fallback_base = _OPENROUTER_URL
    if fallback_base and not fallback_key:
        fallback_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()

    backends: list[tuple[str, str, str]] = [primary]
    if fallback_base and fallback_key and fallback_model:
        backends.append((fallback_base, fallback_key, fallback_model))
    return backends
