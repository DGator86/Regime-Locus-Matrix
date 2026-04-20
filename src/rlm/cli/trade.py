"""``rlm trade`` — generate and execute live/paper trade plans."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from rlm.cli.common import add_backend_arg, add_data_root_arg, add_profile_args, normalize_symbol
from rlm.core.run_manifest import RunManifest, write_run_manifest
from rlm.core.services.trade_service import TradeRequest, TradeService
from rlm.utils.logging import get_run_logger
from rlm.utils.run_id import generate_run_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="rlm trade", description="Generate and optionally execute live/paper trade plans.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--mode", choices=["plan", "paper", "live"], default="plan")
    p.add_argument("--no-kronos", action="store_true")
    p.add_argument("--no-vix", action="store_true")
    p.add_argument("--capital", type=float, default=100_000.0)
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sym = normalize_symbol(args.symbol)
    run_id = generate_run_id("trade")
    log = get_run_logger(__name__, run_id=run_id, command="trade", symbol=sym, backend=args.backend, profile=args.profile)

    req = TradeRequest(symbol=sym, mode=args.mode, use_kronos=not args.no_kronos, attach_vix=not args.no_vix, capital=args.capital, data_root=args.data_root, backend=args.backend)
    result = TradeService().run(req)

    if result.decision:
        print(f"\nTrade decision for {sym}:")
        print(f"  action:   {result.decision.get('roee_action')}")
        print(f"  strategy: {result.decision.get('roee_strategy')}")
        print(f"  size:     {result.decision.get('roee_size_fraction')}")

    manifest = RunManifest(
        run_id=run_id,
        command="trade",
        symbol=sym,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        profile=args.profile,
        config_summary={"mode": args.mode, "use_kronos": not args.no_kronos},
        input_paths={},
        output_paths={},
        metrics={"decision": result.decision or {}, "duration_s": result.duration_s},
    )
    manifest_path = write_run_manifest(manifest, data_root=args.data_root)
    log.info("trade complete", extra={"stage": "complete", "success": True})
    print(f"Run manifest: {manifest_path}")
