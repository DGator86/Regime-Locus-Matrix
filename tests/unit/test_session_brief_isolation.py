from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_session_brief as rsb


def test_session_brief_disables_live_paper_side_effects(monkeypatch, tmp_path: Path) -> None:
    """Inline briefs must not seed the large-options log or rewrite monitor snapshots."""
    captured: dict[str, object] = {}

    def _fake_run(cmd, *, cwd, env_key, default_timeout):  # noqa: ANN001
        captured["cmd"] = list(cmd)
        return 0

    monkeypatch.setattr(rsb, "ROOT", tmp_path)
    monkeypatch.setattr("rlm.utils.subprocess_run.run_with_timeout", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_session_brief.py", "--phase", "preopen", "--top", "3", "--out", "data/processed/session_brief.json"],
    )

    assert rsb.main() == 0
    cmd = [str(x) for x in captured["cmd"]]  # type: ignore[index]
    assert "--no-paper-seed" in cmd
    assert "--no-update-live-side-effects" in cmd
    assert any(str(x).endswith("session_brief.json") for x in cmd)
