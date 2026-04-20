"""``rlm trade`` — generate and execute live/paper trade plans."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlm.cli.common import add_data_root_arg, normalize_symbol
from rlm.core.services.trade_service import TradeRequest, TradeService
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm trade",
        description="Generate and optionally execute live/paper trade plans.",
    )
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g. SPY)")
    p.add_argument(
        "--mode",
        choices=["plan", "paper", "live"],
        default="plan",
        help="Execution mode: plan (print only), paper (IBKR paper), live (IBKR live)",
    )
    p.add_argument("--no-kronos", action="store_true", help="Disable Kronos overlay")
    p.add_argument("--no-vix", action="store_true", help="Skip VIX/VVIX attachment")
    p.add_argument("--capital", type=float, default=100_000.0, help="Account capital")
    p.add_argument("--backend", choices=["auto", "csv", "lake"], default="auto")
    p.add_argument("--profile", default=None, help="Runtime profile name")
    p.add_argument("--config", default=None, help="Path to config JSON")
    p.add_argument("--out-dir", default=None, help="Artifact output directory")
    p.add_argument("--write-artifacts", action=argparse.BooleanOptionalAction, default=True)
    add_data_root_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sym = normalize_symbol(args.symbol)

    log.info("trade start  symbol=%s mode=%s", sym, args.mode)

    req = TradeRequest(
        symbol=sym,
        mode=args.mode,
        use_kronos=not args.no_kronos,
        attach_vix=not args.no_vix,
        capital=args.capital,
        data_root=args.data_root,
        backend=args.backend,
        profile=args.profile,
        config_path=args.config,
        out_dir=None if args.out_dir is None else Path(args.out_dir).expanduser().resolve(),
        write_artifacts=args.write_artifacts,
    )

    result = TradeService().run(req)

    if result.decision:
        print(f"\nTrade decision for {sym}:")
        print(f"  action:   {result.decision.action}")
        print(f"  strategy: {result.decision.strategy}")
        print(f"  size:     {result.decision.size_fraction}")
    if result.executions:
        print("\nExecution log:")
        for entry in result.executions:
            print(f"  mode={entry.mode} success={entry.success} broker={entry.broker} message={entry.message}")
    if result.artifacts.manifest_path:
        print(f"\nArtifacts: {result.artifacts.manifest_path.parent}")
