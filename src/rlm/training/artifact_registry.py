from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ArtifactRegistry:
    active_regime_path: str | None = None
    active_value_path: str | None = None
    previous_regime_path: str | None = None
    previous_value_path: str | None = None
    last_refresh_at: str | None = None


def registry_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "registry.json"


def candidate_dir(base_dir: str | Path, version_tag: str) -> Path:
    return Path(base_dir) / "candidates" / version_tag


def load_registry(base_dir: str | Path) -> ArtifactRegistry:
    path = registry_path(base_dir)
    if not path.exists():
        return ArtifactRegistry()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ArtifactRegistry(**payload)


def save_registry(base_dir: str | Path, registry: ArtifactRegistry) -> None:
    path = registry_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(registry), indent=2), encoding="utf-8")


def make_version_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def active_version_tag(base_dir: str | Path) -> str | None:
    reg = load_registry(base_dir)
    if not reg.active_regime_path:
        return None
    parts = Path(reg.active_regime_path).parts
    if "candidates" in parts:
        idx = parts.index("candidates")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def promote_candidate(
    *,
    base_dir: str | Path,
    candidate_regime_path: Path,
    candidate_value_path: Path,
) -> ArtifactRegistry:
    reg = load_registry(base_dir)
    reg.previous_regime_path = reg.active_regime_path
    reg.previous_value_path = reg.active_value_path
    reg.active_regime_path = str(candidate_regime_path)
    reg.active_value_path = str(candidate_value_path)
    reg.last_refresh_at = datetime.now(timezone.utc).isoformat()
    save_registry(base_dir, reg)
    return reg
