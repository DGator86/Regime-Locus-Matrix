"""``rlm trade`` — generate and execute live/paper trade plans."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlm.cli.common import add_backend_arg, add_data_root_arg, add_profile_args, normalize_symbol
from rlm.core.services.trade_service import TradeRequest, TradeService


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="rlm trade", description="Generate and optionally execute live/paper trade plans.")
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g. SPY)")
    p.add_argument("--mode", choices=["plan", "paper", "live"], default="plan")
    p.add_argument("--no-kronos", action="store_true", help="Disable Kronos overlay")
    p.add_argument("--no-vix", action="store_true", help="Skip VIX/VVIX attachment")
    p.add_argument("--capital", type=float, default=100_000.0, help="Account capital")
    p.add_argument("--out-dir", default=None, help="Artifact output directory")
    p.add_argument("--write-artifacts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--persona",
        action="store_true",
        default=False,
        help="Run the four-stage persona interpretation layer (Seven→Garak→Sisko→Data) and print the result.",
    )
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    req = TradeRequest(
        symbol=normalize_symbol(args.symbol),
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
        use_persona=args.persona,
    )
    result = TradeService().run(req)

    print(f"Trade decision for {req.symbol}:")
    print(f"  action:   {result.decision.action}")
    print(f"  strategy: {result.decision.strategy}")
    print(f"  size:     {result.decision.size_fraction}")
    print("Execution records:")
    for entry in result.executions:
        print(f"  success={entry.success} broker={entry.broker} order_id={entry.order_id} message={entry.message}")
    if result.artifacts.manifest_path:
        print(f"Artifacts: {result.artifacts.manifest_path.parent}")

    if result.persona:
        import json

        print("\nPersona interpretation:")
        print(json.dumps(result.persona.to_dict(), indent=2))
