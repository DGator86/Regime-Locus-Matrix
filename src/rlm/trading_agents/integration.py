"""Adapter wrapping TradingAgentsGraph for use as an RLM Hermes tool.

TradingAgents runs a multi-agent LLM pipeline:
  Analyst Team (market/social/news/fundamentals)
  → Bull/Bear Researcher debate
  → Trader proposal
  → Risk Management debate (aggressive/neutral/conservative)
  → Portfolio Manager final decision (BUY / HOLD / SELL)

The adapter normalises the final PortfolioDecision into a flat dict
that can be serialised to JSON and returned to the Hermes crew.

Provider routing
----------------
By default the adapter auto-detects the best available free LLM provider
(Groq → Google Gemini → OpenRouter → Anthropic → OpenAI).  Groq requires
a small compatibility shim: TradingAgents routes it through the ``openai``
provider with a custom ``backend_url``, but langchain-openai adds
``use_responses_api=True`` for that provider, which Groq does not implement.
``_groq_compat_env`` suppresses that flag for the duration of each
``propagate()`` call and ensures ``OPENAI_API_KEY`` is populated from
``GROQ_API_KEY`` so the request authenticates correctly.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import date as _date
from typing import Any, Dict, Generator, Optional

from rlm.trading_agents.config import _GROQ_BASE_URL, TradingAgentsConfig

log = logging.getLogger(__name__)

# Serializes the monkey-patch window so concurrent Groq analyses don't
# interfere with each other's ChatOpenAI.__init__ replacement.
_groq_patch_lock = threading.Lock()

# Keys tried in order when extracting fields from the decision object/dict.
_ACTION_KEYS = ("action", "final_trade_decision", "trade_decision", "decision")
_RATIONALE_KEYS = ("investment_thesis", "reasoning", "rationale", "summary", "executive_summary")
_ENTRY_KEYS = ("entry_price", "entry", "price_target")
_STOP_KEYS = ("stop_loss", "stop", "stop_price")
_RISK_KEYS = ("risk_level", "risk", "risk_rating")

# ISO-8601 date pattern used to validate analysis_date inputs.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Valid action tokens and their canonical form.
_BUY_TOKENS = frozenset({"BUY", "STRONG BUY", "STRONG_BUY", "OVERWEIGHT"})
_SELL_TOKENS = frozenset({"SELL", "STRONG SELL", "STRONG_SELL", "UNDERWEIGHT"})

# Canonical risk levels and common LLM synonyms.
_RISK_HIGH = frozenset({"HIGH", "HIGH RISK", "HIGH_RISK", "AGGRESSIVE", "VERY HIGH"})
_RISK_LOW = frozenset({"LOW", "LOW RISK", "LOW_RISK", "CONSERVATIVE", "MINIMAL", "VERY LOW"})
_RISK_MODERATE = frozenset({"MODERATE", "MEDIUM", "NEUTRAL", "BALANCED", "NORMAL", "MODERATE RISK"})


@dataclass
class TradingAgentsResult:
    symbol: str
    analysis_date: str
    action: str       # BUY | HOLD | SELL
    rationale: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    risk_level: str   # HIGH | MODERATE | LOW
    confidence: str   # HIGH | MEDIUM | LOW

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pick(obj: Any, keys: tuple, default: Any = None) -> Any:
    """Extract the first non-None matching key from a dict or object."""
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
    """Map raw LLM action tokens to BUY | HOLD | SELL."""
    if raw is None:
        return "HOLD"
    s = str(raw).strip().upper()
    if s in _BUY_TOKENS:
        return "BUY"
    if s in _SELL_TOKENS:
        return "SELL"
    return "HOLD"


def _normalise_risk(raw: Any) -> str:
    """Map raw LLM risk tokens to HIGH | MODERATE | LOW."""
    if raw is None:
        return "MODERATE"
    s = str(raw).strip().upper()
    if not s:
        return "MODERATE"
    if s in _RISK_HIGH:
        return "HIGH"
    if s in _RISK_LOW:
        return "LOW"
    if s in _RISK_MODERATE:
        return "MODERATE"
    # Unknown string: uppercase and pass through so callers can inspect it.
    return s


def _safe_float(value: Any) -> Optional[float]:
    """Parse a numeric value from LLM output, stripping common non-numeric chars."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        cleaned = str(value).replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except (TypeError, ValueError):
            return None


def _derive_confidence(state: dict) -> str:
    """Estimate conviction level from the researcher debate plan in the state.

    Returns HIGH | MEDIUM | LOW:
    - No research plan present          → MEDIUM  (no signal to read)
    - Rating has strong directional word → HIGH   (STRONG / OVER / UNDER)
    - Rating has directional word        → MEDIUM (BUY / SELL without qualifier)
    - Rating field absent or empty       → MEDIUM  (cannot assess)
    - Rating is HOLD / NEUTRAL / other   → LOW    (low-conviction consensus)
    """
    researcher_plan = state.get("research_plan") or state.get("final_research_plan") or {}
    if not researcher_plan:
        return "MEDIUM"
    rating = str(_pick(researcher_plan, ("rating", "score", "confidence"), "") or "").upper().strip()
    if not rating:
        return "MEDIUM"
    if any(w in rating for w in ("STRONG", "OVER", "UNDER")):
        return "HIGH"
    if "BUY" in rating or "SELL" in rating:
        return "MEDIUM"
    return "LOW"


@contextmanager
def _groq_compat_env(groq_key: str) -> Generator[None, None, None]:
    """Shim that makes Groq work with TradingAgents' ``openai`` provider.

    Two problems to solve:
    1. ``OPENAI_API_KEY`` must be set so langchain-openai authenticates against
       Groq's endpoint (Groq accepts the key in the Authorization header, same
       as OpenAI's format).  We always replace it for the duration — including
       when the user already has ``OPENAI_API_KEY`` set to a real OpenAI key.
    2. TradingAgents' ``openai`` provider sets ``use_responses_api=True`` on
       ``ChatOpenAI``, which routes requests to ``/v1/responses`` — an endpoint
       Groq does not implement.  We temporarily replace ``ChatOpenAI.__init__``
       to drop that kwarg for the duration of the ``propagate()`` call.

    The module-level ``_groq_patch_lock`` serialises concurrent Groq analyses
    so the monkey-patch on the global ``ChatOpenAI`` class is never raced.
    """
    saved_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = groq_key

    with _groq_patch_lock:
        _orig = None
        try:
            import langchain_openai as _lo

            _orig = _lo.ChatOpenAI.__init__

            def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                kwargs.pop("use_responses_api", None)
                _orig(self, *args, **kwargs)

            _lo.ChatOpenAI.__init__ = _patched_init
        except (ImportError, AttributeError):
            pass

        try:
            yield
        finally:
            if saved_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = saved_key
            if _orig is not None:
                try:
                    import langchain_openai as _lo  # noqa: F811

                    _lo.ChatOpenAI.__init__ = _orig
                except (ImportError, AttributeError):
                    pass


class TradingAgentsAdapter:
    """Thin wrapper around TradingAgentsGraph that reads config from env vars.

    Lazy-imports tradingagents so the module is importable even when the
    optional dependency is not installed (ImportError raised at call time).
    """

    def __init__(self, config: TradingAgentsConfig | None = None) -> None:
        cfg = config or TradingAgentsConfig.from_env()
        self._cfg = cfg
        self._graph = self._build_graph(cfg)
        self._groq_mode = (
            bool(os.environ.get("GROQ_API_KEY"))
            and (cfg.backend_url or "").rstrip("/") == _GROQ_BASE_URL.rstrip("/")
            and cfg.llm_provider == "openai"
        )
        if self._groq_mode:
            log.info(
                "TradingAgents: using Groq (%s / %s) via openai-compat endpoint",
                cfg.deep_think_llm,
                cfg.quick_think_llm,
            )
        else:
            log.info(
                "TradingAgents: using provider=%s deep=%s quick=%s",
                cfg.llm_provider,
                cfg.deep_think_llm,
                cfg.quick_think_llm,
            )

    @staticmethod
    def _build_graph(cfg: TradingAgentsConfig) -> Any:
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
        ta_config["backend_url"] = cfg.backend_url
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
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must be a non-empty ticker string")

        if analysis_date is None:
            analysis_date = _date.today().strftime("%Y-%m-%d")
        elif not _DATE_RE.match(analysis_date):
            raise ValueError(f"analysis_date must be YYYY-MM-DD, got: {analysis_date!r}")

        log.info(
            "TradingAgents: analyzing %s for %s with analysts=%s",
            symbol,
            analysis_date,
            self._cfg.selected_analysts,
        )

        groq_key = os.environ.get("GROQ_API_KEY", "") if self._groq_mode else ""
        ctx = _groq_compat_env(groq_key) if self._groq_mode and groq_key else nullcontext()

        with ctx:
            state, decision = self._graph.propagate(symbol, analysis_date)

        if not isinstance(state, dict):
            log.debug("TradingAgents: state is %s, not dict — defaulting to {}", type(state).__name__)
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
            entry_price=_safe_float(raw_entry),
            stop_loss=_safe_float(raw_stop),
            risk_level=_normalise_risk(raw_risk),
            confidence=_derive_confidence(state_dict),
        )
