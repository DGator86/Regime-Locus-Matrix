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
