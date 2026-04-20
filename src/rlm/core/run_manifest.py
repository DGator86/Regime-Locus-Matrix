from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rlm.data.paths import get_artifacts_dir


@dataclass
class RunManifest:
    run_id: str
    command: str
    symbol: str
    timestamp_utc: str
    backend: str
    profile: str | None
    config_summary: dict[str, Any]
    input_paths: dict[str, str]
    output_paths: dict[str, str]
    metrics: dict[str, Any]


def write_run_manifest(
    manifest: RunManifest,
    data_root: str | None = None,
    out_path: Path | None = None,
) -> Path:
    if out_path is None:
        runs_dir = get_artifacts_dir(data_root) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{manifest.run_id}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        path = out_path
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return path
