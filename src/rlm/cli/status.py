"""``rlm status`` — consolidated PnL and account status dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from tabulate import tabulate

from rlm.challenge.tracker import ChallengeTracker
from rlm.cli.common import add_data_root_arg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm status",
        description="Consolidated PnL and account status dashboard.",
    )
    add_data_root_arg(p)
    return p.parse_args()


def load_equities_state(data_root: Path) -> dict[str, Any]:
    path = data_root / "processed" / "equity_positions_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_swing_trades(data_root: Path) -> pd.DataFrame:
    path = data_root / "processed" / "trade_log.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root or "data")

    print("\n" + "=" * 60)
    print("  RLM CONSOLIDATED STATUS DASHBOARD  ".center(60))
    print("=" * 60)

    # ---- 1. Daytrade Options (Challenge) ------------------------------------
    print("\n[ DAYTRADE OPTIONS CHALLENGE ]")
    tracker = ChallengeTracker(data_root=str(data_root))
    if tracker.exists():
        state = tracker.load()
        m = state.current_milestone
        bar_filled = int(state.progress_pct * 20)
        bar = "#" * bar_filled + "-" * (20 - bar_filled)
        
        print(f"  Balance   : ${state.balance:>10,.2f}")
        print(f"  Progress  : [{bar}] {state.progress_pct * 100:.1f}% ({getattr(m, 'label', '-')})")
        print(f"  Win Rate  : {state.win_rate:.1%} ({state.wins}W / {state.losses}L)")
        
        if state.open_positions:
            pos_data = []
            for p in state.open_positions:
                pos_data.append([
                    p.symbol, p.option_type.upper(), p.strike, p.qty, p.dte_remaining, f"${p.unrealised_pnl:,.2f}"
                ])
            print("\n  Open Challenge Positions:")
            print(tabulate(pos_data, headers=["Symbol", "Type", "Strike", "Qty", "DTE", "PnL"], tablefmt="simple"))
    else:
        print("  No active challenge found.")

    # ---- 2. Equities --------------------------------------------------------
    print("\n[ EQUITIES ]")
    eq_state = load_equities_state(data_root)
    if eq_state:
        open_eq = [v for v in eq_state.values() if v.get("status") == "open"]
        if open_eq:
            eq_data = []
            for p in open_eq:
                # Calculate simple PnL if possible (this is a placeholder logic as schema is plan-based)
                eq_data.append([
                    p.get("symbol"), p.get("side", "").upper(), p.get("quantity"), p.get("entry_price"), p.get("entry_ts")[:16]
                ])
            print(f"  Open Positions: {len(open_eq)}")
            print(tabulate(eq_data, headers=["Symbol", "Side", "Qty", "Entry", "Opened"], tablefmt="simple"))
        else:
            print("  No open equity positions.")
    else:
        print("  Equity state file not found.")

    # ---- 3. Swing Options ---------------------------------------------------
    print("\n[ SWING OPTIONS ]")
    swing_df = load_swing_trades(data_root)
    if not swing_df.empty:
        # Assuming data/processed/trade_log.csv is for swing trades
        realized = swing_df[swing_df["closed"] == 1]
        unrealized = swing_df[swing_df["closed"] == 0]
        
        total_realized = realized["unrealized_pnl"].sum() if "unrealized_pnl" in realized.columns else 0
        total_unrealized = unrealized["unrealized_pnl"].sum() if "unrealized_pnl" in unrealized.columns else 0
        
        print(f"  Realized PnL   : ${total_realized:>10,.2f}")
        print(f"  Unrealized PnL : ${total_unrealized:>10,.2f}")
        
        if not unrealized.empty:
            print("\n  Open Swing Trades:")
            # Limit to top 5 for brevity
            display_df = unrealized.tail(5)[["symbol", "strategy", "entry_mid", "current_mark", "unrealized_pnl"]]
            print(tabulate(display_df, headers="keys", showindex=False, tablefmt="simple"))
    else:
        print("  No swing trade logs found.")

    print("\n" + "=" * 60)
    print(f"  Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
