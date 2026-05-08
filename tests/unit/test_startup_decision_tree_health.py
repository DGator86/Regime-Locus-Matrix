"""Startup decision-tree health script and snapshot bundle."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_health_snapshot_files_committed() -> None:
    snap = ROOT / "data" / "health_snapshot"
    assert (snap / "manifest.json").is_file()
    assert (snap / "bars_SPY_slice.csv").is_file()
    assert (snap / "spy_large_plan.json").is_file()
    man = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    assert man.get("version") == 1


def test_startup_health_script_exits_zero() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_startup_decision_tree_health.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
