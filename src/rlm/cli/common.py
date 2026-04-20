"""Shared CLI helpers — symbol handling, flags, profiles, and runtime config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rlm.core.config import build_full_config, load_profile, merge_overrides
from rlm.core.pipeline import FullRLMConfig


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def validate_regime_flags(use_hmm: bool, use_markov: bool) -> str:
    if use_hmm and use_markov:
        raise SystemExit("Use either --use-hmm or --use-markov, not both.")
    if use_markov:
        return "markov"
    if use_hmm:
        return "hmm"
    return "hmm"


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-hmm", action="store_true", help="Use HMM regime model (default)")
    parser.add_argument("--hmm-states", type=int, default=None)
    parser.add_argument("--use-markov", action="store_true", help="Use Markov-switching model")
    parser.add_argument("--markov-states", type=int, default=None)
    parser.add_argument("--probabilistic", action="store_true", help="Probabilistic quantile output")
    parser.add_argument("--model-path", default=None, help="Quantile model artifact JSON")
    parser.add_argument("--no-kronos", action="store_true", help="Disable Kronos overlay")
    parser.add_argument("--no-vix", action="store_true", help="Skip VIX/VVIX attachment")


def add_data_root_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--data-root`` argument to *parser*."""
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="DIR",
        help=(
            "Override data root directory (default: RLM_DATA_ROOT env var, "
            "or ./data relative to current working directory)"
        ),
    )


def build_pipeline_config_from_controls(
    *,
    symbol: str,
    profile: str | None,
    config_path: str | None,
    use_kronos: bool,
    attach_vix: bool,
    initial_capital: float,
) -> FullRLMConfig:
    """Build ``FullRLMConfig`` from profile/config controls plus runtime overrides."""
    cfg = FullRLMConfig(symbol=symbol)

    if profile:
        p = profile.lower().strip()
        if p == "live":
            cfg.use_kronos = True
            cfg.attach_vix = True
        elif p == "paper":
            cfg.use_kronos = True
            cfg.attach_vix = False
        elif p == "dev":
            cfg.use_kronos = False
            cfg.attach_vix = False

    if config_path:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        for key, value in payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    cfg.use_kronos = bool(use_kronos)
    cfg.attach_vix = bool(attach_vix)
    cfg.initial_capital = float(initial_capital)
    return cfg
    parser.add_argument("--data-root", default=None, metavar="DIR")


def add_backend_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["auto", "csv", "lake"], default="auto")


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, help="Config profile name from configs/profiles/")
    parser.add_argument("--config", default=None, help="Explicit config file path")


def _cli_overrides(args: argparse.Namespace, symbol: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {"symbol": symbol}

    regime_model = validate_regime_flags(getattr(args, "use_hmm", False), getattr(args, "use_markov", False))
    if getattr(args, "use_hmm", False) or getattr(args, "use_markov", False):
        overrides["regime_model"] = regime_model

    for attr, target in (
        ("hmm_states", "hmm_states"),
        ("markov_states", "markov_states"),
        ("model_path", "probabilistic_model_path"),
        ("initial_capital", "initial_capital"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            overrides[target] = value

    if getattr(args, "probabilistic", False):
        overrides["probabilistic"] = True
    if getattr(args, "no_kronos", False):
        overrides["use_kronos"] = False
    if getattr(args, "no_vix", False):
        overrides["attach_vix"] = False
    if getattr(args, "run_backtest", False):
        overrides["run_backtest"] = True
    return overrides


def build_pipeline_config(
    args: argparse.Namespace,
    symbol: str,
    profile_dict: dict[str, Any] | None = None,
    explicit_overrides: dict[str, Any] | None = None,
) -> FullRLMConfig:
    # defaults
    merged: dict[str, Any] = {}
    # named profile
    if profile_dict:
        merged = merge_overrides(merged, profile_dict)
    # explicit config path or --profile
    if getattr(args, "config", None):
        merged = merge_overrides(merged, load_profile(path=args.config))
    elif getattr(args, "profile", None):
        merged = merge_overrides(merged, load_profile(name=args.profile))
    # explicit call-site overrides
    if explicit_overrides:
        merged = merge_overrides(merged, explicit_overrides)
    # cli wins
    merged = merge_overrides(merged, _cli_overrides(args, symbol))
    return build_full_config(merged)
