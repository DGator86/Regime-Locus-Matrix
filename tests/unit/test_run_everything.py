from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_everything  # noqa: E402


def test_run_everything_passes_force_close_default_to_monitor(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(run_everything, "_run", fake_run)
    monkeypatch.setattr(run_everything.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run_everything.py", "--skip-pipeline"])

    assert run_everything.main() == 0

    monitor_cmd = next(cmd for cmd in commands if cmd[1].endswith("monitor_active_trade_plans.py"))
    force_idx = monitor_cmd.index("--force-close-dte")
    assert monitor_cmd[force_idx + 1] == "0.0"


def test_pipeline_cmd_env_universe_top_and_max_active(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(run_everything, "_run", fake_run)
    monkeypatch.setattr(run_everything.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.delenv("RLM_PIPELINE_ARGS", raising=False)
    monkeypatch.setenv("RLM_UNIVERSE_TOP", "12")
    monkeypatch.setenv("RLM_MAX_ACTIVE_PER_SYMBOL", "2")
    monkeypatch.setattr(sys, "argv", ["run_everything.py", "--skip-monitor"])

    assert run_everything.main() == 0

    pipe = next(c for c in commands if str(c[1]).endswith("run_universe_options_pipeline.py"))
    assert pipe[pipe.index("--top") + 1] == "12"
    assert pipe[pipe.index("--max-active-per-symbol") + 1] == "2"


def test_pipeline_cmd_respects_existing_top_in_rlm_pipeline_args(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(run_everything, "_run", fake_run)
    monkeypatch.setattr(run_everything.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setenv("RLM_PIPELINE_ARGS", "--top 4 --no-vix")
    monkeypatch.setenv("RLM_UNIVERSE_TOP", "99")
    monkeypatch.setattr(sys, "argv", ["run_everything.py", "--skip-monitor"])

    assert run_everything.main() == 0

    pipe = next(c for c in commands if str(c[1]).endswith("run_universe_options_pipeline.py"))
    assert pipe[pipe.index("--top") + 1] == "4"


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = list(cmd)
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0

    def terminate(self):
        return None

    def kill(self):
        return None


def test_initial_pipeline_timeout_still_starts_monitor(monkeypatch, tmp_path) -> None:
    """Timeout/failure of the first universe scan must not skip TP/stop monitoring."""
    plans = tmp_path / "universe_trade_plans.json"
    plans.write_text('{"active_ranked": [{"plan_id": "AAPL_1", "symbol": "AAPL"}]}', encoding="utf-8")
    popens: list[list[str]] = []
    order: list[str] = []

    def fake_pipeline(cmd: list[str]) -> int:
        order.append("pipeline")
        return 124

    def fake_run(cmd: list[str], **kwargs) -> int:
        order.append("run")
        return 0

    def fake_popen(cmd, **kwargs):
        popens.append(list(cmd))
        if str(cmd[1]).endswith("monitor_active_trade_plans.py"):
            order.append("monitor")
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(run_everything, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr(run_everything, "_run", fake_run)
    monkeypatch.setattr(run_everything.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_everything.py",
            "--follow",
            "--paper-trade",
            "--skip-challenge",
            "--rescan-interval",
            "0",
            "--out",
            str(plans),
        ],
    )

    assert run_everything.main() == 0
    monitor_cmds = [c for c in popens if str(c[1]).endswith("monitor_active_trade_plans.py")]
    assert monitor_cmds, "monitor must start after initial pipeline timeout"
    assert order.index("monitor") < order.index("pipeline")


def test_follow_starts_monitor_before_successful_pipeline(monkeypatch, tmp_path) -> None:
    plans = tmp_path / "universe_trade_plans.json"
    plans.write_text('{"active_ranked": []}', encoding="utf-8")
    order: list[str] = []

    def fake_pipeline(cmd: list[str]) -> int:
        order.append("pipeline")
        return 0

    def fake_run(cmd: list[str], **kwargs) -> int:
        return 0

    def fake_popen(cmd, **kwargs):
        if str(cmd[1]).endswith("monitor_active_trade_plans.py"):
            order.append("monitor")
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(run_everything, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr(run_everything, "_run", fake_run)
    monkeypatch.setattr(run_everything.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_everything.py",
            "--follow",
            "--skip-challenge",
            "--rescan-interval",
            "0",
            "--out",
            str(plans),
        ],
    )

    assert run_everything.main() == 0
    assert order[:2] == ["monitor", "pipeline"]


def test_initial_pipeline_failure_without_plans_still_starts_monitor(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "does_not_exist.json"
    popens: list[list[str]] = []

    monkeypatch.setattr(run_everything, "_run_pipeline", lambda cmd: 124)
    monkeypatch.setattr(run_everything, "_run", lambda cmd, **kwargs: 0)

    def fake_popen(cmd, **kwargs):
        popens.append(list(cmd))
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(run_everything.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_everything.py",
            "--follow",
            "--skip-challenge",
            "--rescan-interval",
            "0",
            "--out",
            str(missing),
        ],
    )

    assert run_everything.main() == 0
    assert any(str(c[1]).endswith("monitor_active_trade_plans.py") for c in popens)
