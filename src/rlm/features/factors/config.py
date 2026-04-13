from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from rlm.types.factors import FactorSpec

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


@lru_cache(maxsize=1)
def load_feature_engineering_config() -> dict[str, Any]:
    if not _DEFAULT_CONFIG_PATH.is_file():
        return {}

    loaded = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}

    feature_config = loaded.get("feature_engineering")
    if not isinstance(feature_config, dict):
        return {}
    return feature_config


def _normalize_enabled_factors(raw: object) -> dict[str, set[str]]:
    if not isinstance(raw, dict):
        return {}

    enabled: dict[str, set[str]] = {}
    for category, names in raw.items():
        if not isinstance(category, str):
            continue
        if not isinstance(names, list):
            continue
        enabled[category] = {str(name) for name in names if isinstance(name, str)}
    return enabled


def filter_specs(
    specs: list[FactorSpec],
    feature_config: Mapping[str, object] | None,
) -> list[FactorSpec]:
    if feature_config is None:
        return specs

    enabled_by_category = _normalize_enabled_factors(feature_config.get("enabled_factors"))
    if not enabled_by_category:
        return specs

    filtered: list[FactorSpec] = []
    for spec in specs:
        enabled_names = enabled_by_category.get(spec.category.value)
        if enabled_names is None or spec.name in enabled_names:
            filtered.append(spec)
    return filtered
