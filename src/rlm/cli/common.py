"""Shared CLI helpers — symbol handling, flag validation, config construction.

All CLI sub-commands import from here instead of duplicating logic.
"""

from __future__ import annotations

import argparse

from rlm.core.pipeline import FullRLMConfig
from rlm.roee.engine import ROEEConfig


def normalize_symbol(symbol: str) -> str:
    """Return an upper-cased, stripped ticker symbol."""
    return symbol.strip().upper()


def validate_regime_flags(use_hmm: bool, use_markov: bool) -> str:
    """Validate mutually-exclusive regime flags and return the regime_model string.

    Returns one of ``"hmm"``, ``"markov"``, ``"none"``.

    Raises
    ------
    SystemExit
        When both flags are set simultaneously.
    """
    if use_hmm and use_markov:
        raise SystemExit("Use either --use-hmm or --use-markov, not both.")
    if use_markov:
        return "markov"
    if use_hmm:
        return "hmm"
    return "hmm"  # default when neither flag is set


def build_pipeline_config(args: argparse.Namespace, symbol: str) -> FullRLMConfig:
    """Construct a ``FullRLMConfig`` from parsed CLI args.

    Expects the namespace to contain the standard regime / Kronos / VIX fields
    produced by ``add_pipeline_args()``.
    """
    regime_model = validate_regime_flags(
        getattr(args, "use_hmm", False),
        getattr(args, "use_markov", False),
    )
    return FullRLMConfig(
        symbol=symbol,
        regime_model=regime_model,  # type: ignore[arg-type]
        hmm_states=getattr(args, "hmm_states", 6),
        markov_states=getattr(args, "markov_states", 3),
        probabilistic=getattr(args, "probabilistic", False),
        probabilistic_model_path=getattr(args, "model_path", None),
        use_kronos=not getattr(args, "no_kronos", False),
        attach_vix=not getattr(args, "no_vix", False),
        run_backtest=getattr(args, "run_backtest", False),
        initial_capital=getattr(args, "initial_capital", 100_000.0),
    )


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard regime/forecast/Kronos arguments to *parser*."""
    parser.add_argument("--use-hmm", action="store_true", help="Use HMM regime model (default)")
    parser.add_argument("--hmm-states", type=int, default=6)
    parser.add_argument("--use-markov", action="store_true", help="Use Markov-switching model")
    parser.add_argument("--markov-states", type=int, default=3)
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
