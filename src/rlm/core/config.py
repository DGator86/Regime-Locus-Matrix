"""Config profile loader and pipeline config builder for RLM.

Precedence (highest → lowest):
  1. Explicit CLI flags (passed as *overrides* dict)
  2. Explicit ``--config PATH`` file
  3. Named ``--profile NAME`` (from configs/profiles/)
  4. Package default (configs/default.yaml)

Usage::

    from rlm.core.config import load_profile, build_full_config

    profile = load_profile("backtest")
    cfg = build_full_config(profile, overrides={"hmm_states": 8})
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rlm.utils.logging import get_logger

log = get_logger(__name__)

# Built-in profiles directory (relative to this file's package root)
_PROFILES_DIR = Path(__file__).resolve().parents[3] / "configs" / "profiles"
_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def load_profile(
    name: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a config profile dict.

    Parameters
    ----------
    name:
        Named profile (e.g. ``"default"``, ``"forecast"``, ``"backtest"``, ``"live"``).
        Loads from ``configs/profiles/<name>.yaml``.
    path:
        Explicit path to a YAML config file.  Takes precedence over *name*.

    Returns
    -------
    dict
        Merged config dict (base defaults + profile overrides).
    """
    import yaml  # pyyaml; always in base deps

    # Start from the package default
    base: dict[str, Any] = {}
    if _DEFAULT_CONFIG.is_file():
        with open(_DEFAULT_CONFIG, encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
        log.debug("config loaded default  path=%s", _DEFAULT_CONFIG)

    # Explicit file path takes precedence
    if path is not None:
        explicit_path = Path(path).expanduser().resolve()
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Config file not found: {explicit_path}")
        with open(explicit_path, encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        log.info("config loaded explicit  path=%s", explicit_path)
        return _deep_merge(base, override)

    # Named profile
    if name is not None:
        profile_path = _PROFILES_DIR / f"{name}.yaml"
        if not profile_path.is_file():
            available = [p.stem for p in _PROFILES_DIR.glob("*.yaml")] if _PROFILES_DIR.exists() else []
            raise ValueError(
                f"Unknown profile: {name!r}\n"
                f"Available profiles: {available}"
            )
        with open(profile_path, encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        log.info("config loaded profile  name=%s path=%s", name, profile_path)
        return _deep_merge(base, override)

    log.debug("config using defaults only (no profile or path specified)")
    return base


def build_full_config(
    profile: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> "FullRLMConfig":
    """Build a ``FullRLMConfig`` from a profile dict with optional overrides.

    Parameters
    ----------
    profile:
        Profile dict returned by ``load_profile()``.
    overrides:
        CLI flag values (or any key=value pairs) that take highest precedence.
        Only keys that map to ``FullRLMConfig`` fields are applied.

    Returns
    -------
    FullRLMConfig
    """
    from rlm.core.pipeline import FullRLMConfig
    from rlm.roee.engine import ROEEConfig

    merged = _deep_merge(profile, overrides or {})

    forecast = merged.get("forecast", {})
    regime = merged.get("regime", {})
    roee = merged.get("roee_hmm", {})
    backtest = merged.get("backtest", {})
    kronos = merged.get("kronos", {})

    regime_model = regime.get("model", "hmm")

    roee_config = ROEEConfig(
        use_dynamic_sizing=roee.get("use_dynamic_sizing", False),
        max_kelly_fraction=roee.get("max_kelly_fraction", 0.25),
        max_capital_fraction=roee.get("max_capital_fraction", 0.5),
        vol_target=roee.get("vol_target", 0.15),
        hmm_confidence_threshold=roee.get("hmm_confidence_threshold", 0.6),
        transition_penalty=roee.get("transition_penalty", 0.5),
        regime_adjusted_kelly=roee.get("regime_adjusted_kelly", True),
    )

    cfg = FullRLMConfig(
        regime_model=regime_model,  # type: ignore[arg-type]
        hmm_states=regime.get("hmm_states", 6),
        markov_states=regime.get("markov_k_regimes", 3),
        drift_gamma_alpha=forecast.get("drift_gamma_alpha", 0.65),
        sigma_floor=forecast.get("sigma_floor", 1e-4),
        direction_neutral_threshold=forecast.get("direction_neutral_threshold", 0.3),
        move_window=forecast.get("move_window", 100),
        vol_window=forecast.get("vol_window", 100),
        use_kronos=kronos.get("enabled", True),
        run_backtest=backtest.get("run_backtest", False),
        initial_capital=backtest.get("initial_capital", 100_000.0),
        roee_config=roee_config,
    )

    log.debug("config built  regime=%s kronos=%s backtest=%s", regime_model, cfg.use_kronos, cfg.run_backtest)
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: *base* deep-merged with *override* (override wins)."""
    result = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def list_profiles() -> list[str]:
    """Return available profile names from the profiles directory."""
    if not _PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))
