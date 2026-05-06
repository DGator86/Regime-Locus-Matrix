"""Resolve OpenAI-compatible LLM endpoints for the Hermes crew."""

from __future__ import annotations

import json
import os
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

_OPENROUTER_URL = "https://openrouter.ai/api/v1"
# OpenRouter free-tier models use the `:free` suffix (quota subject to their policy).
_DEFAULT_OPENROUTER_FREE_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
_DEFAULT_OLLAMA_MODEL = "llama3.2"
_DEFAULT_OLLAMA_MIN_CONTEXT = 64000
_DEFAULT_OLLAMA_PREFERRED_MODELS = (
    "qwen3",
    "qwen2.5",
    "qwen2",
    "qwen",
    "llama3.3",
    "llama3.2",
    "phi4",
    "phi3",
    "mistral",
)


def _truthy(key: str) -> bool:
    return (os.environ.get(key) or "").strip().lower() in ("1", "true", "yes", "on")


def _is_local_ollama_base(base_url: str) -> bool:
    parsed = urlsplit(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost"}


def _ollama_tags_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/api/tags"


def _preferred_ollama_prefixes() -> tuple[str, ...]:
    raw = (os.environ.get("RLM_HERMES_OLLAMA_MODEL_PREFER") or "").strip()
    if not raw:
        return _DEFAULT_OLLAMA_PREFERRED_MODELS
    parts = [x.strip().lower() for x in raw.split(",")]
    out = tuple(x for x in parts if x)
    return out or _DEFAULT_OLLAMA_PREFERRED_MODELS


def _select_preferred_ollama_model(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    lowered = [c.lower() for c in candidates]
    for pref in _preferred_ollama_prefixes():
        for idx, item in enumerate(lowered):
            if item.startswith(pref):
                return candidates[idx]
    return candidates[0]


def _ollama_show_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/api/show"


def _ollama_context_length(base_url: str, model: str) -> int | None:
    timeout = float((os.environ.get("RLM_HERMES_OLLAMA_TAGS_TIMEOUT_SEC") or "2.0").strip() or "2.0")
    payload = {"model": model}
    try:
        from urllib.request import Request

        request = Request(
            _ollama_show_url(base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=max(0.2, timeout)) as resp:  # nosec B310
            raw = json.loads(resp.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return None

    info = raw.get("model_info")
    if not isinstance(info, dict):
        return None
    for key in ("llama.context_length", "qwen2.context_length", "qwen35.context_length", "context_length"):
        val = info.get(key)
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return None


def _ollama_min_context() -> int:
    raw = (os.environ.get("RLM_HERMES_OLLAMA_MIN_CONTEXT") or "").strip()
    if not raw:
        return _DEFAULT_OLLAMA_MIN_CONTEXT
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_OLLAMA_MIN_CONTEXT


def _detect_ollama_model(base_url: str) -> str:
    tags_url = _ollama_tags_url(base_url)
    timeout = float((os.environ.get("RLM_HERMES_OLLAMA_TAGS_TIMEOUT_SEC") or "2.0").strip() or "2.0")
    try:
        with urlopen(tags_url, timeout=max(0.2, timeout)) as resp:  # nosec B310
            payload = json.loads(resp.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return _DEFAULT_OLLAMA_MODEL
    models = payload.get("models")
    if not isinstance(models, list):
        return _DEFAULT_OLLAMA_MODEL
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    if not names:
        return _DEFAULT_OLLAMA_MODEL

    min_ctx = _ollama_min_context()
    if min_ctx <= 0:
        return _select_preferred_ollama_model(names) or _DEFAULT_OLLAMA_MODEL

    by_model_context: dict[str, int | None] = {name: _ollama_context_length(base_url, name) for name in names}
    eligible = [name for name in names if (by_model_context.get(name) or 0) >= min_ctx]
    if eligible:
        return _select_preferred_ollama_model(eligible) or eligible[0]
    return _select_preferred_ollama_model(names) or _DEFAULT_OLLAMA_MODEL


def _resolve_model(base_url: str, default_model: str) -> str:
    env_model = (os.environ.get("RLM_HERMES_MODEL") or os.environ.get("LLM_MODEL") or "").strip()
    if env_model:
        return env_model
    if _is_local_ollama_base(base_url):
        return _detect_ollama_model(base_url)
    return default_model


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
        model = _resolve_model(explicit_base, _DEFAULT_OLLAMA_MODEL)
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
        local_base = "http://127.0.0.1:11434/v1"
        model = _resolve_model(local_base, _DEFAULT_OLLAMA_MODEL)
        primary = (local_base, "ollama", model)

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
