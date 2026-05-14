from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path

import pandas as pd


def test_health_check_default_output_does_not_modify_live_gate(monkeypatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    live_gate = repo_root / "data" / "processed" / "gate_state.json"
    previous = live_gate.read_text(encoding="utf-8") if live_gate.exists() else None
    sentinel = {
        "posture": "STAND-DOWN",
        "status": "CRITICAL",
        "last_updated": "2099-01-01T00:00:00Z",
    }

    pipeline_mod = types.ModuleType("rlm.core.pipeline")

    class FullRLMConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FullRLMPipeline:
        def __init__(self, cfg: FullRLMConfig) -> None:
            self.cfg = cfg

        def run(self, bars: pd.DataFrame) -> object:
            return types.SimpleNamespace(
                factors_df=pd.DataFrame({"S_D": [0.1]}),
                forecast_df=pd.DataFrame(
                    {
                        "forecast_return": [0.01],
                        "hmm_confidence": [0.8],
                    }
                ),
                policy_df=pd.DataFrame(
                    {
                        "roee_action": ["hold"],
                        "roee_size_fraction": [0.0],
                        "hmm_state": [2],
                    }
                ),
            )

    pipeline_mod.FullRLMConfig = FullRLMConfig
    pipeline_mod.FullRLMPipeline = FullRLMPipeline

    health_mod = types.ModuleType("rlm.hermes_facts.health")
    health_roots: list[Path] = []

    def gather_health_report(root: Path) -> dict[str, object]:
        health_roots.append(root)
        return {
            "overall_ok": True,
            "report_text": "[Health report]\n  Overall: OK",
        }

    health_mod.gather_health_report = gather_health_report

    market_mod = types.ModuleType("rlm.hermes_facts.market_context")
    market_roots: list[Path] = []

    def build_trade_and_regime_context(root: Path) -> str:
        market_roots.append(root)
        return "Market State: test\nOVERALL RISK POSTURE: LOW\nActive plans: 3"

    market_mod.build_trade_and_regime_context = build_trade_and_regime_context

    loop_mod = types.ModuleType("rlm.hermes_crew.loop")
    loop_roots: list[Path] = []
    loop_mod._load_commander_skill_text = lambda root: "commander skill text for isolated health check"
    loop_mod._load_pipeline_health_skill_text = lambda root: "pipeline health skill stub"
    loop_mod._load_regime_research_skill_text = lambda root: "regime research skill stub"

    def _agent_response(root: Path, *_args: object) -> str:
        loop_roots.append(root)
        return "agent response"

    def _run_full_briefing(root: Path, *_args: object) -> tuple[str, str, str]:
        loop_roots.append(root)
        return (
            "pipeline health report",
            "regime research report",
            "SYSTEM STATUS: NOMINAL\n"
            "MARKET POSTURE: NORMAL\n"
            "COMMAND DECISION: HOLD\n"
            "RATIONALE: synthetic isolated health check\n"
            "CREW ORDERS:\n"
            "  - Pipeline Health: maintain current status\n"
            "  - Regime Research: continue monitoring\n"
            "  - Trading Engine: hold\n",
        )

    loop_mod._run_pipeline_health_agent = _agent_response
    loop_mod._run_regime_research_agent = _agent_response
    loop_mod._run_full_briefing = _run_full_briefing

    monkeypatch.setitem(sys.modules, "rlm.core.pipeline", pipeline_mod)
    monkeypatch.setitem(sys.modules, "rlm.hermes_facts.health", health_mod)
    monkeypatch.setitem(sys.modules, "rlm.hermes_facts.market_context", market_mod)
    monkeypatch.setitem(sys.modules, "rlm.hermes_crew.loop", loop_mod)
    monkeypatch.setitem(sys.modules, "run_agent", types.ModuleType("run_agent"))
    monkeypatch.setitem(
        sys.modules,
        "rlm_hermes_tools.register_rlm_tools",
        types.ModuleType("rlm_hermes_tools.register_rlm_tools"),
    )

    isolated_root = tmp_path / "health-check-output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/rlm_health_check.py", "--output-root", str(isolated_root)],
    )

    live_gate.parent.mkdir(parents=True, exist_ok=True)
    live_gate.write_text(json.dumps(sentinel, indent=2), encoding="utf-8")
    try:
        try:
            runpy.run_path(str(repo_root / "scripts" / "rlm_health_check.py"), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 0

        assert json.loads(live_gate.read_text(encoding="utf-8")) == sentinel
        assert (isolated_root / "data" / "processed" / "gate_state.json").is_file()
        assert (isolated_root / "data" / "artifacts" / "crew_decisions.json").is_file()
        assert health_roots and all(root == isolated_root for root in health_roots)
        assert market_roots and all(root == isolated_root for root in market_roots)
        assert loop_roots and all(root == isolated_root for root in loop_roots)
    finally:
        if previous is None:
            live_gate.unlink(missing_ok=True)
        else:
            live_gate.write_text(previous, encoding="utf-8")
