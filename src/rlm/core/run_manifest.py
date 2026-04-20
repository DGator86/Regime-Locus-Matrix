"""Run manifest — per-invocation audit record for every RLM command.

Every command (forecast, backtest, ingest, trade) generates a ``RunManifest``
that is written to ``<data-root>/artifacts/runs/<run_id>.json``.  This ties
logs, configs, inputs, and outputs together for reproducibility and debugging.

Usage::

    from rlm.core.run_manifest import RunManifest, new_run_id

    run_id = new_run_id()
    manifest = RunManifest(
        run_id=run_id,
        command="forecast",
        symbol="SPY",
        ...
    )
    path = manifest.write(data_root="/mnt/lake")
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlm.data.paths import get_data_root
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def new_run_id() -> str:
    """Return a short, URL-safe run identifier (8 hex chars)."""
    return uuid.uuid4().hex[:8]


@dataclass
class RunManifest:
    """Audit record for a single RLM command invocation."""

    run_id: str
    command: str
    symbol: str
    timestamp_utc: str
    config_summary: dict[str, Any] = field(default_factory=dict)
    input_paths: dict[str, str] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    backend: str = "auto"
    profile: str | None = None
    success: bool = True
    error: str | None = None
    duration_s: float = 0.0

    def write(self, data_root: str | Path | None = None) -> Path:
        """Persist manifest JSON to ``<data-root>/artifacts/runs/<run_id>.json``."""
        root = get_data_root(data_root)
        runs_dir = root / "artifacts" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        path = runs_dir / f"{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        log.info("manifest written  run_id=%s path=%s", self.run_id, path)
        return path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, run_id: str, data_root: str | Path | None = None) -> "RunManifest":
        """Load a manifest by run ID from the artifacts store."""
        root = get_data_root(data_root)
        path = root / "artifacts" / "runs" / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"No manifest for run_id={run_id!r}: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**raw)


def build_config_summary(cfg: Any) -> dict[str, Any]:
    """Extract a minimal reproducibility summary from ``FullRLMConfig``."""
    return {
        "regime_model": getattr(cfg, "regime_model", None),
        "hmm_states": getattr(cfg, "hmm_states", None),
        "use_kronos": getattr(cfg, "use_kronos", None),
        "probabilistic": getattr(cfg, "probabilistic", None),
        "run_backtest": getattr(cfg, "run_backtest", None),
        "initial_capital": getattr(cfg, "initial_capital", None),
        "attach_vix": getattr(cfg, "attach_vix", None),
    }


def now_utc() -> str:
    """Return ISO-8601 UTC timestamp string."""
    return datetime.now(tz=timezone.utc).isoformat()
