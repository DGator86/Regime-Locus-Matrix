from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _auto_detect_provider() -> Tuple[str, str, str, Optional[str]]:
    """Return (provider, deep_model, quick_model, backend_url) from available API keys.

    Priority order: Groq (already in RLM) → Google Gemini (free AI Studio) →
    OpenRouter (free-tier models) → Anthropic → OpenAI.
    All options except Anthropic/OpenAI are free-tier accessible.
    """
    if os.environ.get("GROQ_API_KEY"):
        # Groq is OpenAI-compatible; TradingAgents routes it through the openai
        # provider with a custom backend_url.  The adapter suppresses the
        # use_responses_api flag (Groq does not implement that endpoint).
        return (
            "openai",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            _GROQ_BASE_URL,
        )
    if os.environ.get("GOOGLE_API_KEY"):
        # Google AI Studio free tier — native TradingAgents support.
        return ("google", "gemini-2.0-flash", "gemini-2.0-flash", None)
    if os.environ.get("OPENROUTER_API_KEY"):
        # OpenRouter free-tier models (no per-token cost for :free variants).
        return (
            "openrouter",
            "meta-llama/llama-4-maverick:free",
            "google/gemini-2.0-flash-exp:free",
            None,
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("anthropic", "claude-opus-4-7", "claude-haiku-4-5-20251001", None)
    # Fallback: standard OpenAI (paid)
    return ("openai", "gpt-4o", "gpt-4o-mini", None)


@dataclass
class TradingAgentsConfig:
    """Configuration for the TradingAgents multi-agent analysis pipeline.

    All fields read from environment variables so they can be set in .env
    alongside the existing RLM / Hermes configuration.

    When no TRADING_AGENTS_LLM_PROVIDER is set, the provider is auto-detected
    from available API keys (Groq → Google → OpenRouter → Anthropic → OpenAI).
    """

    llm_provider: str = "openai"
    deep_think_llm: str = "llama-3.3-70b-versatile"
    quick_think_llm: str = "llama-3.1-8b-instant"
    backend_url: Optional[str] = _GROQ_BASE_URL
    max_debate_rounds: int = 1
    max_risk_discuss_rounds: int = 1
    selected_analysts: List[str] = field(
        default_factory=lambda: ["market", "news", "fundamentals"]
    )
    online_tools: bool = False
    checkpoint_enabled: bool = False

    @classmethod
    def from_env(cls) -> "TradingAgentsConfig":
        explicit_provider = os.environ.get("TRADING_AGENTS_LLM_PROVIDER", "").strip()
        explicit_deep = os.environ.get("TRADING_AGENTS_DEEP_THINK_LLM", "").strip()
        explicit_quick = os.environ.get("TRADING_AGENTS_QUICK_THINK_LLM", "").strip()
        explicit_backend = os.environ.get("TRADING_AGENTS_BACKEND_URL", "").strip() or None

        if explicit_provider:
            provider = explicit_provider
            deep = explicit_deep or "llama-3.3-70b-versatile"
            quick = explicit_quick or "llama-3.1-8b-instant"
            backend = explicit_backend
        else:
            provider, deep, quick, backend = _auto_detect_provider()
            if explicit_deep:
                deep = explicit_deep
            if explicit_quick:
                quick = explicit_quick
            if explicit_backend:
                backend = explicit_backend

        analysts_raw = os.environ.get("TRADING_AGENTS_ANALYSTS", "market,news,fundamentals")
        analysts = [a.strip() for a in analysts_raw.split(",") if a.strip()]

        return cls(
            llm_provider=provider,
            deep_think_llm=deep,
            quick_think_llm=quick,
            backend_url=backend,
            max_debate_rounds=_parse_int_env("TRADING_AGENTS_MAX_DEBATE_ROUNDS", 1),
            max_risk_discuss_rounds=_parse_int_env("TRADING_AGENTS_MAX_RISK_ROUNDS", 1),
            selected_analysts=analysts,
            online_tools=os.environ.get("TRADING_AGENTS_ONLINE_TOOLS", "0").lower() in ("1", "true", "yes"),
            checkpoint_enabled=os.environ.get("TRADING_AGENTS_CHECKPOINT", "0").lower() in ("1", "true", "yes"),
        )
