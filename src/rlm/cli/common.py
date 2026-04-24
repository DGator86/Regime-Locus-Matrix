"""Shared CLI helpers — symbol handling, flags, profiles, and runtime config."""

from __future__ import annotations

import argparse
from typing import Any

from rlm.core.config import build_pipeline_config as build_core_pipeline_config
from rlm.core.pipeline import FullRLMConfig
from rlm.data.liquidity_universe import EXPANDED_LIQUID_UNIVERSE


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def resolve_backtest_symbols(args: argparse.Namespace) -> list[str]:
    """Resolve ``--symbol`` (default), ``--symbols``, or ``--universe`` into a ticker list."""
    if getattr(args, "universe", False) and getattr(args, "symbols", None):
        raise SystemExit("Use either --universe or --symbols, not both.")
    if getattr(args, "universe", False):
        return list(EXPANDED_LIQUID_UNIVERSE)
    if getattr(args, "symbols", None):
        out = [normalize_symbol(x) for x in str(args.symbols).split(",") if x.strip()]
        if not out:
            raise SystemExit("--symbols must list at least one ticker.")
        return out
    return [normalize_symbol(args.symbol)]


def validate_regime_flags(use_hmm: bool, use_markov: bool) -> str:
    if use_hmm and use_markov:
        raise SystemExit("Use either --use-hmm or --use-markov, not both.")
    if use_markov:
        return "markov"
    return "hmm"


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-hmm", action="store_true", help="Use HMM regime model (default)")
    parser.add_argument("--hmm-states", type=int, default=None)
    parser.add_argument("--use-markov", action="store_true", help="Use Markov-switching model")
    parser.add_argument("--markov-states", type=int, default=None)
    parser.add_argument(
        "--probabilistic", action="store_true", help="Probabilistic quantile output"
    )
    parser.add_argument("--model-path", default=None, help="Quantile model artifact JSON")
    parser.add_argument("--no-kronos", action="store_true", help="Disable Kronos overlay")
    parser.add_argument("--no-vix", action="store_true", help="Skip VIX/VVIX attachment")


def add_data_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="DIR",
        help="Override data root directory (default: RLM_DATA_ROOT env var, else ./data)",
    )


def add_backend_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["auto", "csv", "lake"], default="auto")


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile", default=None, help="Config profile name from configs/profiles/"
    )
    parser.add_argument("--config", default=None, help="Explicit config file path")


def _cli_overrides(args: argparse.Namespace, symbol: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {"symbol": symbol}
    regime_model = validate_regime_flags(
        getattr(args, "use_hmm", False), getattr(args, "use_markov", False)
    )
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
    merged_overrides = _cli_overrides(args, symbol)
    if profile_dict:
        merged_overrides = {**profile_dict, **merged_overrides}
    if explicit_overrides:
        merged_overrides = {**merged_overrides, **explicit_overrides}
    return build_core_pipeline_config(
        symbol=symbol,
        profile=getattr(args, "profile", None),
        config_path=getattr(args, "config", None),
        overrides=merged_overrides,
    )
