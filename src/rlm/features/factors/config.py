from __future__ import annotations

import os
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


# Extra scored factors for short-DTE / SPY daytrade path (candles + MTF confluence + S/R).
_SHORT_DTE_FACTOR_OVERLAY: dict[str, list[str]] = {
    "direction": [
        "candle_bullish_reversal",
        "candle_bearish_reversal",
        "candle_continuation",
        "doji_or_spinning_top",
        "mtf_confluencema_spread_over_atr",
        "mtf_confluencero_c_n",
        "dist_to_nearest_support",
        "dist_to_nearest_resistance",
        "breakout_confirmed",
    ],
}

_KRONOS_FACTOR_NAMES = {
    "kronos_return_forecast",
    "kronos_range_forecast",
    "kronos_path_dispersion",
}


def feature_config_for_pipeline(
    *,
    short_dte: bool = False,
    include_kronos: bool = True,
) -> dict[str, Any]:
    """Base YAML config, optionally merged with short-DTE scoring overlay."""
    base = dict(load_feature_engineering_config())
    use_overlay = short_dte or (os.environ.get("RLM_SHORT_DTE_SCORING") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    enabled = base.get("enabled_factors")
    if not isinstance(enabled, dict):
        enabled = {}
    merged: dict[str, list[str]] = {}
    for cat, names in enabled.items():
        if isinstance(names, list):
            merged[str(cat)] = list(names)
    if use_overlay:
        for cat, extra in _SHORT_DTE_FACTOR_OVERLAY.items():
            merged.setdefault(cat, [])
            for name in extra:
                if name not in merged[cat]:
                    merged[cat].append(name)
    if not include_kronos:
        for cat, names in list(merged.items()):
            merged[cat] = [name for name in names if name not in _KRONOS_FACTOR_NAMES]
    base["enabled_factors"] = merged
    return base


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
