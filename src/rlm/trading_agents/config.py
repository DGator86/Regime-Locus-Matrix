from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class TradingAgentsConfig:
    """Configuration for the TradingAgents multi-agent analysis pipeline.

    All fields read from environment variables so they can be set in .env
    alongside the existing RLM / Hermes configuration.
    """

    llm_provider: str = "anthropic"
    deep_think_llm: str = "claude-opus-4-7"
    quick_think_llm: str = "claude-haiku-4-5-20251001"
    max_debate_rounds: int = 1
    max_risk_discuss_rounds: int = 1
    selected_analysts: List[str] = field(
        default_factory=lambda: ["market", "news", "fundamentals"]
    )
    online_tools: bool = False
    checkpoint_enabled: bool = False

    @classmethod
    def from_env(cls) -> "TradingAgentsConfig":
        analysts_raw = os.environ.get("TRADING_AGENTS_ANALYSTS", "market,news,fundamentals")
        analysts = [a.strip() for a in analysts_raw.split(",") if a.strip()]
        return cls(
            llm_provider=os.environ.get("TRADING_AGENTS_LLM_PROVIDER", "anthropic"),
            deep_think_llm=os.environ.get("TRADING_AGENTS_DEEP_THINK_LLM", "claude-opus-4-7"),
            quick_think_llm=os.environ.get("TRADING_AGENTS_QUICK_THINK_LLM", "claude-haiku-4-5-20251001"),
            max_debate_rounds=int(os.environ.get("TRADING_AGENTS_MAX_DEBATE_ROUNDS", "1")),
            max_risk_discuss_rounds=int(os.environ.get("TRADING_AGENTS_MAX_RISK_ROUNDS", "1")),
            selected_analysts=analysts,
            online_tools=os.environ.get("TRADING_AGENTS_ONLINE_TOOLS", "0").lower() in ("1", "true", "yes"),
            checkpoint_enabled=os.environ.get("TRADING_AGENTS_CHECKPOINT", "0").lower() in ("1", "true", "yes"),
        )
