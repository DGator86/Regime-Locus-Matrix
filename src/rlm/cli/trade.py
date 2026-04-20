"""``rlm trade`` — generate and execute live/paper trade plans."""

from __future__ import annotations

import argparse

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
    )

    result = TradeService().run(req)

    if result.decision:
        print(f"\nTrade decision for {sym}:")
        print(f"  action:   {result.decision.get('roee_action')}")
        print(f"  strategy: {result.decision.get('roee_strategy')}")
        print(f"  size:     {result.decision.get('roee_size_fraction')}")
    if result.execution_log:
        print("\nExecution log:")
        for entry in result.execution_log:
            print(f"  {entry}")
