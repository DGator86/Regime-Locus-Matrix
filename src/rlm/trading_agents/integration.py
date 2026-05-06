"""Adapter wrapping TradingAgentsGraph for use as an RLM Hermes tool.

TradingAgents runs a multi-agent LLM pipeline:
  Analyst Team (market/social/news/fundamentals)
  → Bull/Bear Researcher debate
  → Trader proposal
  → Risk Management debate (aggressive/neutral/conservative)
  → Portfolio Manager final decision (BUY / HOLD / SELL)

The adapter normalises the final PortfolioDecision into a flat dict
that can be serialised to JSON and returned to the Hermes crew.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date as _date
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Keys tried in order when extracting the action from the decision object/dict.
_ACTION_KEYS = ("action", "final_trade_decision", "trade_decision", "decision")
_RATIONALE_KEYS = ("investment_thesis", "reasoning", "rationale", "summary", "executive_summary")
_ENTRY_KEYS = ("entry_price", "entry", "price_target")
_STOP_KEYS = ("stop_loss", "stop", "stop_price")
_RISK_KEYS = ("risk_level", "risk", "risk_rating")


@dataclass
class TradingAgentsResult:
    symbol: str
    analysis_date: str
    action: str  # BUY / HOLD / SELL
    rationale: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    risk_level: str
    confidence: str  # HIGH / MEDIUM / LOW derived from consensus

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pick(obj: Any, keys: tuple, default: Any = None) -> Any:
    """Extract the first matching key from a dict or object."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
    else:
        for k in keys:
            v = getattr(obj, k, None)
            if v is not None:
                return v
    return default


def _normalise_action(raw: Any) -> str:
    if raw is None:
        return "HOLD"
    s = str(raw).strip().upper()
    if s in ("BUY", "OVERWEIGHT", "STRONG BUY"):
        return "BUY"
    if s in ("SELL", "UNDERWEIGHT", "STRONG SELL"):
        return "SELL"
    return "HOLD"


def _derive_confidence(state: dict) -> str:
    """Estimate conviction from debate agreement in the final state."""
    researcher_plan = state.get("research_plan") or state.get("final_research_plan", {})
    if not researcher_plan:
        return "MEDIUM"
    rating = str(_pick(researcher_plan, ("rating", "score", "confidence"), "")).upper()
    if "BUY" in rating or "SELL" in rating:
        return "HIGH" if "STRONG" in rating or "OVER" in rating or "UNDER" in rating else "MEDIUM"
    return "LOW"


class TradingAgentsAdapter:
    """Thin wrapper around TradingAgentsGraph that reads config from env vars.

    Lazy-imports tradingagents so the module is importable even when the
    optional dependency is not installed (ImportError raised at call time).
    """

    def __init__(self, config: "TradingAgentsConfig | None" = None) -> None:
        from rlm.trading_agents.config import TradingAgentsConfig as _Cfg

        cfg = config or _Cfg.from_env()
        self._cfg = cfg
        self._graph = self._build_graph(cfg)

    @staticmethod
    def _build_graph(cfg: "TradingAgentsConfig") -> Any:
        try:
            from tradingagents.default_config import DEFAULT_CONFIG
            from tradingagents.graph.trading_graph import TradingAgentsGraph
        except ImportError as exc:
            raise ImportError(
                "tradingagents is not installed. "
                "Run: pip install -e '.[trading_agents]'"
            ) from exc

        ta_config = DEFAULT_CONFIG.copy()
        ta_config["llm_provider"] = cfg.llm_provider
        ta_config["deep_think_llm"] = cfg.deep_think_llm
        ta_config["quick_think_llm"] = cfg.quick_think_llm
        ta_config["max_debate_rounds"] = cfg.max_debate_rounds
        ta_config["max_risk_discuss_rounds"] = cfg.max_risk_discuss_rounds
        ta_config["online_tools"] = cfg.online_tools
        ta_config["checkpoint_enabled"] = cfg.checkpoint_enabled

        return TradingAgentsGraph(
            selected_analysts=cfg.selected_analysts,
            debug=False,
            config=ta_config,
        )

    def analyze(self, symbol: str, analysis_date: str | None = None) -> TradingAgentsResult:
        if analysis_date is None:
            analysis_date = _date.today().strftime("%Y-%m-%d")

        log.info("TradingAgents: analyzing %s for %s with analysts=%s", symbol, analysis_date, self._cfg.selected_analysts)

        state, decision = self._graph.propagate(symbol, analysis_date)
        state_dict: dict = state if isinstance(state, dict) else {}

        raw_action = _pick(decision, _ACTION_KEYS, _pick(state_dict, _ACTION_KEYS, "HOLD"))
        raw_rationale = _pick(decision, _RATIONALE_KEYS, _pick(state_dict, _RATIONALE_KEYS, ""))
        raw_entry = _pick(decision, _ENTRY_KEYS, _pick(state_dict, _ENTRY_KEYS))
        raw_stop = _pick(decision, _STOP_KEYS, _pick(state_dict, _STOP_KEYS))
        raw_risk = _pick(decision, _RISK_KEYS, _pick(state_dict, _RISK_KEYS, "MODERATE"))

        return TradingAgentsResult(
            symbol=symbol,
            analysis_date=analysis_date,
            action=_normalise_action(raw_action),
            rationale=str(raw_rationale or "").strip(),
            entry_price=float(raw_entry) if raw_entry is not None else None,
            stop_loss=float(raw_stop) if raw_stop is not None else None,
            risk_level=str(raw_risk or "MODERATE").upper(),
            confidence=_derive_confidence(state_dict),
        )
