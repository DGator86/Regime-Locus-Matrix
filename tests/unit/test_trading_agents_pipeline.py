"""Full pipeline health check for the TradingAgents integration.

Tests every stage with synthetic matching data:
  Stage 1 — Config loading (_auto_detect_provider, TradingAgentsConfig.from_env)
  Stage 2 — Data normalizers (_normalise_action, _normalise_risk, _safe_float,
             _derive_confidence, _pick)
  Stage 3 — Adapter.analyze() with various synthetic TradingAgents output formats
  Stage 4 — TradingAgentsResult serialisation (to_dict completeness & types)
  Stage 5 — Hermes fact gatherer (gather_trading_agents_analysis response contract)
  Stage 6 — Tool handler (register_rlm_tools JSON schema & error shapes)

Synthetic data intentionally covers:
  - Dict-format decisions (most common TradingAgents output)
  - Object-format decisions (Pydantic model attributes)
  - Partial decisions (missing optional fields)
  - LLM price strings ("$524.50", "$1,234.00", "N/A")
  - All action token variants
  - All risk level variants
  - All confidence derivation paths
"""
from __future__ import annotations

import json
from typing import Any

import pytest

import importlib.util
import pathlib

import rlm.trading_agents.integration as integration_mod

# Load the fact module directly from its file to avoid triggering
# rlm.hermes_facts.__init__, which eagerly imports market_context → pandas.
_fact_path = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src" / "rlm" / "hermes_facts" / "trading_agents_analysis.py"
)
_spec = importlib.util.spec_from_file_location("rlm.hermes_facts.trading_agents_analysis", _fact_path)
ta_module = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(ta_module)  # type: ignore[union-attr]
from rlm.trading_agents.config import (
    _GROQ_BASE_URL,
    _PROVIDER_DEFAULTS,
    _auto_detect_provider,
    _parse_int_env,
)
from rlm.trading_agents.integration import (
    TradingAgentsResult,
    _derive_confidence,
    _normalise_action,
    _normalise_risk,
    _pick,
    _safe_float,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic TradingAgents output payloads
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_FULL_DICT = {
    "final_trade_decision": "BUY",
    "investment_thesis": (
        "SPY shows strong bullish momentum with confirmed breakout above 200-day MA. "
        "Fundamentals support continued rally into Q3."
    ),
    "entry_price": "$524.50",
    "stop_loss": "$1,510.00",  # deliberately has comma
    "risk_level": "moderate",  # lowercase — must normalise to MODERATE
    "price_target": "540.00",
}

SYNTHETIC_FULL_STATE = {
    "research_plan": {
        "rating": "OVERWEIGHT",
        "recommendation": "Bull thesis confirmed by 3/4 analysts",
    },
    "messages": [{"role": "assistant", "content": "Analysis complete."}],
}

# Object-style decision (simulates a Pydantic model)
class _ObjDecision:
    action = "SELL"
    investment_thesis = "Bearish regime: risk-off posture recommended."
    entry_price = 498.00
    stop_loss = 510.00
    risk_level = "HIGH"


# Minimal decision — only required field
SYNTHETIC_MINIMAL_DICT = {"action": "HOLD"}

# Decision with "N/A" prices (LLM couldn't produce price levels)
SYNTHETIC_NO_PRICES_DICT = {
    "action": "BUY",
    "investment_thesis": "Directional conviction without price target.",
    "entry_price": "N/A",
    "stop_loss": "N/A",
    "risk_level": "Low",
}

SYNTHETIC_MINIMAL_STATE: dict = {}

_ALL_ENV_KEYS = (
    "GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
)


def _clear_provider_keys(monkeypatch):
    for k in _ALL_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Config loading
# ─────────────────────────────────────────────────────────────────────────────

class TestStage1Config:
    def test_default_config_has_no_backend_url(self):
        """TradingAgentsConfig() bare default must NOT preset a Groq URL."""
        from rlm.trading_agents.config import TradingAgentsConfig
        cfg = TradingAgentsConfig()
        assert cfg.backend_url is None

    def test_from_env_groq_sets_backend_url(self, monkeypatch):
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        from rlm.trading_agents.config import TradingAgentsConfig
        cfg = TradingAgentsConfig.from_env()
        assert cfg.backend_url == _GROQ_BASE_URL
        assert cfg.llm_provider == "openai"
        assert "llama" in cfg.deep_think_llm

    def test_from_env_google_no_backend_url(self, monkeypatch):
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza_test")
        from rlm.trading_agents.config import TradingAgentsConfig
        cfg = TradingAgentsConfig.from_env()
        assert cfg.backend_url is None
        assert cfg.llm_provider == "google"
        assert "gemini" in cfg.deep_think_llm

    def test_from_env_openrouter_no_backend_url(self, monkeypatch):
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        from rlm.trading_agents.config import TradingAgentsConfig
        cfg = TradingAgentsConfig.from_env()
        assert cfg.backend_url is None
        assert cfg.llm_provider == "openrouter"

    def test_explicit_anthropic_provider_uses_correct_models(self, monkeypatch):
        """TRADING_AGENTS_LLM_PROVIDER=anthropic must not default to Llama models."""
        monkeypatch.setenv("TRADING_AGENTS_LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("TRADING_AGENTS_DEEP_THINK_LLM", raising=False)
        monkeypatch.delenv("TRADING_AGENTS_QUICK_THINK_LLM", raising=False)
        from rlm.trading_agents.config import TradingAgentsConfig
        cfg = TradingAgentsConfig.from_env()
        assert cfg.llm_provider == "anthropic"
        assert "claude" in cfg.deep_think_llm.lower()
        assert "claude" in cfg.quick_think_llm.lower()
        assert cfg.backend_url is None

    def test_explicit_google_provider_uses_correct_models(self, monkeypatch):
        monkeypatch.setenv("TRADING_AGENTS_LLM_PROVIDER", "google")
        monkeypatch.delenv("TRADING_AGENTS_DEEP_THINK_LLM", raising=False)
        monkeypatch.delenv("TRADING_AGENTS_QUICK_THINK_LLM", raising=False)
        from rlm.trading_agents.config import TradingAgentsConfig
        cfg = TradingAgentsConfig.from_env()
        assert cfg.llm_provider == "google"
        assert "gemini" in cfg.deep_think_llm.lower()

    def test_explicit_model_overrides_provider_default(self, monkeypatch):
        monkeypatch.setenv("TRADING_AGENTS_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("TRADING_AGENTS_DEEP_THINK_LLM", "claude-opus-4-7")
        monkeypatch.setenv("TRADING_AGENTS_QUICK_THINK_LLM", "claude-sonnet-4-6")
        from rlm.trading_agents.config import TradingAgentsConfig
        cfg = TradingAgentsConfig.from_env()
        assert cfg.deep_think_llm == "claude-opus-4-7"
        assert cfg.quick_think_llm == "claude-sonnet-4-6"

    def test_explicit_backend_url_overrides_auto_detect(self, monkeypatch):
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("TRADING_AGENTS_BACKEND_URL", "http://localhost:11434/v1")
        from rlm.trading_agents.config import TradingAgentsConfig
        cfg = TradingAgentsConfig.from_env()
        assert cfg.backend_url == "http://localhost:11434/v1"

    def test_analysts_parsed_correctly(self, monkeypatch):
        monkeypatch.setenv("TRADING_AGENTS_ANALYSTS", "market, social , news")
        from rlm.trading_agents.config import TradingAgentsConfig
        cfg = TradingAgentsConfig.from_env()
        assert cfg.selected_analysts == ["market", "social", "news"]

    def test_provider_defaults_covers_common_providers(self):
        for provider in ("openai", "anthropic", "google", "openrouter", "deepseek"):
            assert provider in _PROVIDER_DEFAULTS, f"Missing provider default for {provider}"
            deep, quick = _PROVIDER_DEFAULTS[provider]
            assert deep and quick, f"Empty model name for {provider}"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Data normalizers
# ─────────────────────────────────────────────────────────────────────────────

class TestStage2Normalizers:

    # ── _normalise_action ────────────────────────────────────────────────────
    @pytest.mark.parametrize("raw,expected", [
        ("BUY", "BUY"),
        ("buy", "BUY"),             # lowercase
        ("OVERWEIGHT", "BUY"),
        ("STRONG BUY", "BUY"),
        ("STRONG_BUY", "BUY"),
        ("SELL", "SELL"),
        ("sell", "SELL"),
        ("UNDERWEIGHT", "SELL"),
        ("STRONG SELL", "SELL"),
        ("STRONG_SELL", "SELL"),
        ("HOLD", "HOLD"),
        ("hold", "HOLD"),
        (None, "HOLD"),
        ("UNKNOWN", "HOLD"),        # unrecognised → HOLD
        ("", "HOLD"),               # empty string → HOLD
        (123, "HOLD"),              # numeric → HOLD (not a token)
        ({"action": "BUY"}, "HOLD"),  # dict as raw → str("{'action': 'BUY'}") → HOLD
    ])
    def test_normalise_action(self, raw, expected):
        assert _normalise_action(raw) == expected

    # ── _normalise_risk ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("raw,expected", [
        ("HIGH", "HIGH"),
        ("high", "HIGH"),
        ("HIGH RISK", "HIGH"),
        ("AGGRESSIVE", "HIGH"),
        ("LOW", "LOW"),
        ("low", "LOW"),
        ("CONSERVATIVE", "LOW"),
        ("MINIMAL", "LOW"),
        ("MODERATE", "MODERATE"),
        ("MEDIUM", "MODERATE"),
        ("NEUTRAL", "MODERATE"),
        ("BALANCED", "MODERATE"),
        (None, "MODERATE"),
        ("", "MODERATE"),
        ("UNKNOWN_LEVEL", "UNKNOWN_LEVEL"),  # passthrough unknown
    ])
    def test_normalise_risk(self, raw, expected):
        assert _normalise_risk(raw) == expected

    # ── _safe_float ──────────────────────────────────────────────────────────
    @pytest.mark.parametrize("value,expected", [
        (524.50, 524.50),
        ("524.50", 524.50),
        ("$524.50", 524.50),
        ("$1,510.00", 1510.00),
        ("$1,234,567.89", 1234567.89),
        (0, 0.0),
        (None, None),
        ("N/A", None),
        ("n/a", None),
        ("--", None),
        ("", None),
        ("not a number", None),
    ])
    def test_safe_float(self, value, expected):
        assert _safe_float(value) == expected

    # ── _pick ────────────────────────────────────────────────────────────────
    def test_pick_from_dict_first_key_wins(self):
        d = {"a": 1, "b": 2}
        assert _pick(d, ("a", "b")) == 1

    def test_pick_from_dict_skips_none(self):
        d = {"a": None, "b": "value"}
        assert _pick(d, ("a", "b")) == "value"

    def test_pick_from_dict_returns_default_when_all_missing(self):
        assert _pick({}, ("x", "y"), "default") == "default"

    def test_pick_from_object_reads_attributes(self):
        obj = _ObjDecision()
        result = _pick(obj, ("action", "final_trade_decision"))
        assert result == "SELL"

    def test_pick_from_object_skips_none_attributes(self):
        class _Obj:
            action = None
            investment_thesis = "thesis"
        assert _pick(_Obj(), ("action", "investment_thesis")) == "thesis"

    def test_pick_falsy_zero_is_valid(self):
        d = {"entry_price": 0.0}
        assert _pick(d, ("entry_price",)) == 0.0

    def test_pick_from_none_obj_returns_default(self):
        # obj=None is not a dict, getattr works but returns None for all keys
        assert _pick(None, ("a", "b"), "fallback") == "fallback"

    # ── _derive_confidence ───────────────────────────────────────────────────
    def test_derive_confidence_no_state(self):
        assert _derive_confidence({}) == "MEDIUM"

    def test_derive_confidence_missing_research_plan(self):
        assert _derive_confidence({"messages": []}) == "MEDIUM"

    def test_derive_confidence_overweight_is_high(self):
        state = {"research_plan": {"rating": "OVERWEIGHT"}}
        assert _derive_confidence(state) == "HIGH"

    def test_derive_confidence_underweight_is_high(self):
        state = {"research_plan": {"rating": "UNDERWEIGHT"}}
        assert _derive_confidence(state) == "HIGH"

    def test_derive_confidence_strong_buy_is_high(self):
        state = {"research_plan": {"rating": "STRONG BUY"}}
        assert _derive_confidence(state) == "HIGH"

    def test_derive_confidence_strong_sell_is_high(self):
        state = {"research_plan": {"rating": "STRONG SELL"}}
        assert _derive_confidence(state) == "HIGH"

    def test_derive_confidence_buy_without_qualifier_is_medium(self):
        state = {"research_plan": {"rating": "BUY"}}
        assert _derive_confidence(state) == "MEDIUM"

    def test_derive_confidence_sell_without_qualifier_is_medium(self):
        state = {"research_plan": {"rating": "SELL"}}
        assert _derive_confidence(state) == "MEDIUM"

    def test_derive_confidence_hold_is_low(self):
        state = {"research_plan": {"rating": "HOLD"}}
        assert _derive_confidence(state) == "LOW"

    def test_derive_confidence_empty_rating_is_medium(self):
        state = {"research_plan": {"rating": ""}}
        assert _derive_confidence(state) == "MEDIUM"

    def test_derive_confidence_missing_rating_field_is_medium(self):
        state = {"research_plan": {"recommendation": "no rating key"}}
        assert _derive_confidence(state) == "MEDIUM"

    def test_derive_confidence_uses_final_research_plan_key(self):
        state = {"final_research_plan": {"rating": "OVERWEIGHT"}}
        assert _derive_confidence(state) == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Adapter.analyze() with synthetic decision formats
# ─────────────────────────────────────────────────────────────────────────────

def _make_adapter_with_mock(monkeypatch, state, decision):
    """Build a TradingAgentsAdapter whose _graph.propagate returns (state, decision)."""
    from rlm.trading_agents.config import TradingAgentsConfig

    class _MockGraph:
        def propagate(self, symbol, date):
            return state, decision

    from rlm.trading_agents import integration as _int_mod

    cfg = TradingAgentsConfig(
        llm_provider="openai",
        deep_think_llm="gpt-4o",
        quick_think_llm="gpt-4o-mini",
        backend_url=None,
    )
    adapter = object.__new__(_int_mod.TradingAgentsAdapter)
    adapter._cfg = cfg
    adapter._graph = _MockGraph()
    adapter._groq_mode = False
    return adapter


class TestStage3Adapter:

    def test_full_dict_decision_parsed_correctly(self, monkeypatch):
        adapter = _make_adapter_with_mock(monkeypatch, SYNTHETIC_FULL_STATE, SYNTHETIC_FULL_DICT)
        result = adapter.analyze("SPY", "2026-05-06")
        assert result.action == "BUY"
        assert result.entry_price == pytest.approx(524.50)
        assert result.stop_loss == pytest.approx(1510.00)
        assert result.risk_level == "MODERATE"
        assert result.confidence == "HIGH"   # OVERWEIGHT in research_plan
        assert result.symbol == "SPY"
        assert result.analysis_date == "2026-05-06"
        assert "bullish momentum" in result.rationale

    def test_object_decision_parsed_correctly(self, monkeypatch):
        adapter = _make_adapter_with_mock(monkeypatch, SYNTHETIC_MINIMAL_STATE, _ObjDecision())
        result = adapter.analyze("QQQ", "2026-05-06")
        assert result.action == "SELL"
        assert result.entry_price == pytest.approx(498.00)
        assert result.stop_loss == pytest.approx(510.00)
        assert result.risk_level == "HIGH"
        assert result.confidence == "MEDIUM"   # no research plan → MEDIUM

    def test_minimal_dict_defaults_correctly(self, monkeypatch):
        adapter = _make_adapter_with_mock(monkeypatch, {}, SYNTHETIC_MINIMAL_DICT)
        result = adapter.analyze("IWM", "2026-05-06")
        assert result.action == "HOLD"
        assert result.entry_price is None
        assert result.stop_loss is None
        assert result.risk_level == "MODERATE"
        assert result.confidence == "MEDIUM"
        assert result.rationale == ""

    def test_na_prices_become_none(self, monkeypatch):
        adapter = _make_adapter_with_mock(monkeypatch, {}, SYNTHETIC_NO_PRICES_DICT)
        result = adapter.analyze("SPY", "2026-05-06")
        assert result.action == "BUY"
        assert result.entry_price is None
        assert result.stop_loss is None
        assert result.risk_level == "LOW"

    def test_non_dict_state_handled_gracefully(self, monkeypatch):
        """state returned as a list should not crash — defaults to empty dict."""
        adapter = _make_adapter_with_mock(monkeypatch, ["unexpected", "list"], SYNTHETIC_MINIMAL_DICT)
        result = adapter.analyze("SPY", "2026-05-06")
        assert result.confidence == "MEDIUM"   # no research_plan found

    def test_symbol_uppercased_automatically(self, monkeypatch):
        adapter = _make_adapter_with_mock(monkeypatch, {}, SYNTHETIC_MINIMAL_DICT)
        result = adapter.analyze("spy", "2026-05-06")
        assert result.symbol == "SPY"

    def test_empty_symbol_raises(self, monkeypatch):
        adapter = _make_adapter_with_mock(monkeypatch, {}, SYNTHETIC_MINIMAL_DICT)
        with pytest.raises(ValueError, match="non-empty"):
            adapter.analyze("", "2026-05-06")

    def test_invalid_date_format_raises(self, monkeypatch):
        adapter = _make_adapter_with_mock(monkeypatch, {}, SYNTHETIC_MINIMAL_DICT)
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            adapter.analyze("SPY", "2026/05/06")

    def test_date_defaults_to_today(self, monkeypatch):
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        adapter = _make_adapter_with_mock(monkeypatch, {}, SYNTHETIC_MINIMAL_DICT)
        result = adapter.analyze("SPY")
        assert result.analysis_date == today

    def test_action_from_state_when_missing_in_decision(self, monkeypatch):
        """Fallback chain: decision has no action → read from state_dict."""
        state = {"action": "SELL", "research_plan": {"rating": "SELL"}}
        decision = {"investment_thesis": "Bears confirmed."}  # no action key
        adapter = _make_adapter_with_mock(monkeypatch, state, decision)
        result = adapter.analyze("SPY", "2026-05-06")
        assert result.action == "SELL"

    def test_rationale_from_state_when_missing_in_decision(self, monkeypatch):
        state = {"investment_thesis": "State-level thesis."}
        decision = {"action": "BUY"}  # no rationale key
        adapter = _make_adapter_with_mock(monkeypatch, state, decision)
        result = adapter.analyze("SPY", "2026-05-06")
        assert result.rationale == "State-level thesis."


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — TradingAgentsResult serialisation
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_RESULT_KEYS = frozenset({
    "symbol", "analysis_date", "action", "rationale",
    "entry_price", "stop_loss", "risk_level", "confidence",
})


class TestStage4Serialisation:

    def _make_result(self, **overrides) -> TradingAgentsResult:
        defaults: dict[str, Any] = dict(
            symbol="SPY",
            analysis_date="2026-05-06",
            action="BUY",
            rationale="Test rationale.",
            entry_price=520.0,
            stop_loss=510.0,
            risk_level="MODERATE",
            confidence="HIGH",
        )
        return TradingAgentsResult(**{**defaults, **overrides})

    def test_to_dict_contains_all_required_keys(self):
        d = self._make_result().to_dict()
        assert EXPECTED_RESULT_KEYS.issubset(d.keys())

    def test_to_dict_no_unexpected_keys(self):
        d = self._make_result().to_dict()
        assert set(d.keys()) == EXPECTED_RESULT_KEYS

    def test_to_dict_is_json_serialisable(self):
        d = self._make_result().to_dict()
        serialised = json.dumps(d)
        assert json.loads(serialised) == d

    def test_to_dict_none_prices_serialise_as_null(self):
        d = self._make_result(entry_price=None, stop_loss=None).to_dict()
        assert d["entry_price"] is None
        assert d["stop_loss"] is None
        serialised = json.dumps(d)
        assert '"entry_price": null' in serialised

    def test_to_dict_types_are_correct(self):
        d = self._make_result().to_dict()
        assert isinstance(d["symbol"], str)
        assert isinstance(d["analysis_date"], str)
        assert isinstance(d["action"], str)
        assert isinstance(d["rationale"], str)
        assert isinstance(d["entry_price"], float)
        assert isinstance(d["stop_loss"], float)
        assert isinstance(d["risk_level"], str)
        assert isinstance(d["confidence"], str)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Hermes fact gatherer response contract
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_RESPONSE_KEYS = frozenset({
    "available", "symbol", "analysis_date", "action", "rationale",
    "entry_price", "stop_loss", "risk_level", "confidence", "error",
})


class TestStage5HermesFact:

    def _assert_contract(self, result: dict, symbol: str):
        """Every response path must include the full key set."""
        missing = EXPECTED_RESPONSE_KEYS - result.keys()
        assert not missing, f"Missing keys in response: {missing}"
        assert result["symbol"] == symbol

    def test_import_error_returns_full_shape(self, monkeypatch):
        class _BrokenAdapter:
            def __init__(self, *a, **kw):
                raise ImportError("not installed")

        monkeypatch.setattr(integration_mod, "TradingAgentsAdapter", _BrokenAdapter)
        result = ta_module.gather_trading_agents_analysis("SPY")
        self._assert_contract(result, "SPY")
        assert result["available"] is False
        assert result["error"] is not None
        assert result["analysis_date"] is not None   # must always be present

    def test_runtime_error_returns_full_shape(self, monkeypatch):
        class _FlakyAdapter:
            def __init__(self, *a, **kw):
                pass
            def analyze(self, *a, **kw):
                raise RuntimeError("LLM timeout")

        monkeypatch.setattr(integration_mod, "TradingAgentsAdapter", _FlakyAdapter)
        result = ta_module.gather_trading_agents_analysis("NVDA", "2026-05-06")
        self._assert_contract(result, "NVDA")
        assert result["available"] is False
        assert "LLM timeout" in result["error"]
        assert result["analysis_date"] == "2026-05-06"

    def test_success_returns_full_shape_with_data(self, monkeypatch):
        mock_result = TradingAgentsResult(
            symbol="SPY", analysis_date="2026-05-06", action="BUY",
            rationale="Strong breakout.", entry_price=524.0, stop_loss=510.0,
            risk_level="MODERATE", confidence="HIGH",
        )

        class _GoodAdapter:
            def __init__(self, *a, **kw):
                pass
            def analyze(self, symbol, date=None):
                return mock_result

        monkeypatch.setattr(integration_mod, "TradingAgentsAdapter", _GoodAdapter)
        result = ta_module.gather_trading_agents_analysis("SPY", "2026-05-06")
        self._assert_contract(result, "SPY")
        assert result["available"] is True
        assert result["error"] is None
        assert result["action"] == "BUY"

    def test_error_data_fields_are_none(self, monkeypatch):
        class _BrokenAdapter:
            def __init__(self, *a, **kw):
                raise RuntimeError("fail")

        monkeypatch.setattr(integration_mod, "TradingAgentsAdapter", _BrokenAdapter)
        result = ta_module.gather_trading_agents_analysis("SPY")
        for field in ("action", "rationale", "entry_price", "stop_loss", "risk_level", "confidence"):
            assert result[field] is None, f"Expected {field}=None on error, got {result[field]!r}"

    def test_date_default_populated_on_error(self, monkeypatch):
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")

        class _BrokenAdapter:
            def __init__(self, *a, **kw):
                raise RuntimeError("fail")

        monkeypatch.setattr(integration_mod, "TradingAgentsAdapter", _BrokenAdapter)
        result = ta_module.gather_trading_agents_analysis("SPY")
        assert result["analysis_date"] == today


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — Tool handler (register_rlm_tools)
# ─────────────────────────────────────────────────────────────────────────────

def _stub_tools_registry(monkeypatch):
    """Stub out tools.registry so register_rlm_tools can be imported without hermes-agent."""
    import sys
    from types import ModuleType

    if "tools" not in sys.modules:
        tools_mod = ModuleType("tools")
        monkeypatch.setitem(sys.modules, "tools", tools_mod)

    if "tools.registry" not in sys.modules:
        reg_mod = ModuleType("tools.registry")

        class _FakeRegistry:
            def register(self, **kwargs):
                pass

        reg_mod.registry = _FakeRegistry()
        monkeypatch.setitem(sys.modules, "tools.registry", reg_mod)

    # Stub Hermes fact imports pulled in at module level.
    # Each stub must expose the names that register_rlm_tools imports from it.

    # Ensure parent packages are present and marked as packages (need __path__).
    for parent in ("rlm.hermes_facts", "rlm.roee"):
        if parent not in sys.modules:
            pkg = ModuleType(parent)
            pkg.__path__ = []  # marks it as a package so submodule imports work
            monkeypatch.setitem(sys.modules, parent, pkg)
        else:
            # If already registered (could be the real package), nothing to do.
            pass

    _dep_attrs: dict = {
        "rlm.hermes_facts.health": {"gather_health_report": lambda *a, **kw: {}},
        "rlm.hermes_facts.market_context": {"build_trade_and_regime_context": lambda *a, **kw: ""},
        # Point to the already-loaded ta_module so gather_trading_agents_analysis resolves.
        "rlm.hermes_facts.trading_agents_analysis": {
            "gather_trading_agents_analysis": ta_module.gather_trading_agents_analysis,
        },
        "rlm.roee.system_gate": {"SystemGate": type("SystemGate", (), {
            "__init__": lambda self, *a, **kw: None,
            "load": lambda self: type("St", (), {"posture": "", "status": "", "last_updated": ""})(),
            "is_trading_allowed": lambda self: False,
        })},
    }

    for dep, attrs in _dep_attrs.items():
        if dep not in sys.modules:
            stub = ModuleType(dep)
            for name, val in attrs.items():
                setattr(stub, name, val)
            monkeypatch.setitem(sys.modules, dep, stub)


class TestStage6ToolHandler:

    def _get_handler(self, monkeypatch=None):
        if monkeypatch is not None:
            _stub_tools_registry(monkeypatch)
        # Force re-import now that stubs are in place
        import importlib
        import sys
        sys.modules.pop("rlm_hermes_tools.register_rlm_tools", None)
        sys.modules.pop("rlm_hermes_tools", None)
        mod = importlib.import_module("rlm_hermes_tools.register_rlm_tools")
        return mod._rlm_get_trading_agents_analysis_json

    def test_missing_symbol_returns_consistent_error_shape(self, monkeypatch):
        handler = self._get_handler(monkeypatch)
        raw = handler({})
        result = json.loads(raw)
        assert result["available"] is False
        assert result["symbol"] == ""
        assert "error" in result

    def test_empty_symbol_returns_error(self, monkeypatch):
        handler = self._get_handler(monkeypatch)
        raw = handler({"symbol": "  "})
        result = json.loads(raw)
        assert result["available"] is False

    def test_symbol_is_uppercased(self, monkeypatch):
        handler = self._get_handler(monkeypatch)
        monkeypatch.setattr(
            ta_module,
            "gather_trading_agents_analysis",
            lambda symbol, date=None: {"available": True, "symbol": symbol, "action": "HOLD",
                                       "analysis_date": "2026-05-06", "rationale": "",
                                       "entry_price": None, "stop_loss": None,
                                       "risk_level": "MODERATE", "confidence": "MEDIUM",
                                       "error": None},
        )
        raw = handler({"symbol": "spy"})
        result = json.loads(raw)
        assert result["symbol"] == "SPY"

    def test_date_param_forwarded(self, monkeypatch):
        handler = self._get_handler(monkeypatch)
        captured: list = []

        def _fake(symbol, date=None):
            captured.append((symbol, date))
            return {"available": False, "symbol": symbol, "analysis_date": date,
                    "action": None, "rationale": None, "entry_price": None,
                    "stop_loss": None, "risk_level": None, "confidence": None, "error": "test"}

        import sys
        reg_mod = sys.modules["rlm_hermes_tools.register_rlm_tools"]
        monkeypatch.setattr(reg_mod, "gather_trading_agents_analysis", _fake)
        handler({"symbol": "SPY", "date": "2026-05-06"})
        assert captured[0] == ("SPY", "2026-05-06")

    def test_analysis_date_alias_also_works(self, monkeypatch):
        handler = self._get_handler(monkeypatch)
        captured: list = []

        def _fake(symbol, date=None):
            captured.append((symbol, date))
            return {"available": False, "symbol": symbol, "analysis_date": date,
                    "action": None, "rationale": None, "entry_price": None,
                    "stop_loss": None, "risk_level": None, "confidence": None, "error": "test"}

        import sys
        reg_mod = sys.modules["rlm_hermes_tools.register_rlm_tools"]
        monkeypatch.setattr(reg_mod, "gather_trading_agents_analysis", _fake)
        handler({"symbol": "SPY", "analysis_date": "2026-05-06"})
        assert captured[0] == ("SPY", "2026-05-06")

    def test_handler_output_is_valid_json(self, monkeypatch):
        handler = self._get_handler(monkeypatch)

        class _BrokenAdapter:
            def __init__(self, *a, **kw):
                raise RuntimeError("test failure")

        monkeypatch.setattr(integration_mod, "TradingAgentsAdapter", _BrokenAdapter)
        raw = handler({"symbol": "SPY"})
        parsed = json.loads(raw)  # must not raise
        assert isinstance(parsed, dict)

    def test_schema_requires_symbol(self, monkeypatch):
        _stub_tools_registry(monkeypatch)
        import importlib, sys
        sys.modules.pop("rlm_hermes_tools.register_rlm_tools", None)
        sys.modules.pop("rlm_hermes_tools", None)
        mod = importlib.import_module("rlm_hermes_tools.register_rlm_tools")
        assert "symbol" in mod.RLM_TRADING_AGENTS_SCHEMA["parameters"]["required"]

    def test_schema_date_is_optional(self, monkeypatch):
        _stub_tools_registry(monkeypatch)
        import importlib, sys
        sys.modules.pop("rlm_hermes_tools.register_rlm_tools", None)
        sys.modules.pop("rlm_hermes_tools", None)
        mod = importlib.import_module("rlm_hermes_tools.register_rlm_tools")
        params = mod.RLM_TRADING_AGENTS_SCHEMA["parameters"]
        assert "date" in params["properties"]
        assert "date" not in params.get("required", [])
