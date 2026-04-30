from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_health_check_output_root_does_not_overwrite_live_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    live_gate = repo_root / "data" / "processed" / "gate_state.json"
    original_text = live_gate.read_text(encoding="utf-8") if live_gate.exists() else None
    sentinel = {
        "posture": "STAND-DOWN",
        "status": "CRITICAL",
        "last_updated": "1999-01-01T00:00:00Z",
    }

    live_gate.parent.mkdir(parents=True, exist_ok=True)
    live_gate.write_text(json.dumps(sentinel, indent=2), encoding="utf-8")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "rlm_health_check.py"),
                "--output-root",
                str(tmp_path / "isolated"),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(live_gate.read_text(encoding="utf-8")) == sentinel

        isolated_root = tmp_path / "isolated" / "data"
        assert (isolated_root / "processed" / "gate_state.json").is_file()
        assert (isolated_root / "artifacts" / "crew_decisions.json").is_file()
    finally:
        if original_text is None:
            live_gate.unlink(missing_ok=True)
        else:
            live_gate.write_text(original_text, encoding="utf-8")
