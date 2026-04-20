from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rlm.core.pipeline import FullRLMConfig


_DEFAULT_PROFILES_DIR = Path(__file__).resolve().parents[3] / "configs" / "profiles"


def load_profile(name: str | None = None, path: str | Path | None = None) -> dict[str, Any]:
    if path is not None:
        profile_path = Path(path).expanduser().resolve()
    else:
        profile_name = name or "default"
        profile_path = _DEFAULT_PROFILES_DIR / f"{profile_name}.yaml"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile/config file not found: {profile_path}")
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config at {profile_path}: expected mapping")
    return data


def merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_full_config(profile: dict[str, Any]) -> FullRLMConfig:
    return FullRLMConfig(**{k: v for k, v in profile.items() if k in FullRLMConfig.__annotations__})


def build_pipeline_config(
    *,
    symbol: str,
    profile: str | None = None,
    config_path: str | Path | None = None,
    initial_capital: float | None = None,
    overrides: dict[str, Any] | None = None,
) -> FullRLMConfig:
    merged: dict[str, Any] = {}
    if profile:
        merged = merge_overrides(merged, load_profile(name=profile))
    if config_path:
        merged = merge_overrides(merged, load_profile(path=config_path))

    merged = merge_overrides(merged, {"symbol": symbol})
    merged = merge_overrides(
        merged,
        {
            "symbol": symbol,
            "use_kronos": bool(use_kronos),
            "attach_vix": bool(attach_vix),
        },
    )
    if initial_capital is not None:
        merged["initial_capital"] = float(initial_capital)
    if overrides:
        merged = merge_overrides(merged, overrides)
    return build_full_config(merged)
