"""``rlm challenge`` — $1K→$25K aggressive options dry-run challenge.

Runs in complete isolation from the standard IBKR equities / options flow.
State lives under ``data/challenge/``.  No real orders are ever placed.

Commands
--------
``rlm challenge --reset``
    Wipe state and start fresh at $1,000 (or ``--capital``).

``rlm challenge --run``
    Run one session: evaluates open positions, then considers a new entry
    based on the persona pipeline directive.

``rlm challenge --status``
    Print current balance, open positions, milestone progress, and stats.
"""

from __future__ import annotations

import argparse
import json
import sys

from rlm.challenge.config import MILESTONES, ChallengeConfig
from rlm.challenge.engine import ChallengeEngine
from rlm.challenge.tracker import ChallengeTracker
from rlm.cli.common import add_backend_arg, add_data_root_arg, normalize_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm challenge",
        description="$1K→$25K aggressive options dry-run challenge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  rlm challenge --reset --symbol SPY\n"
            "  rlm challenge --run   --symbol SPY\n"
            "  rlm challenge --status\n"
        ),
    )
    p.add_argument("--symbol", default="SPY", help="Underlying ticker (default: SPY)")
    p.add_argument("--reset", action="store_true", help="Reset challenge state to seed capital")
    p.add_argument("--run", action="store_true", help="Run one session (loads data + persona)")
    p.add_argument("--status", action="store_true", help="Print challenge dashboard")
    p.add_argument(
        "--capital",
        type=float,
        default=1_000.0,
        help="Seed capital for --reset (default: $1,000)",
    )
    p.add_argument(
        "--target",
        type=float,
        default=25_000.0,
        help="Target capital (default: $25,000)",
    )
    p.add_argument(
        "--underlying-price",
        type=float,
        default=None,
        help="Override underlying price (skips live data fetch for --run)",
    )
    p.add_argument(
        "--iv",
        type=float,
        default=None,
        help="Override implied volatility, e.g. 0.18 for 18%% (default: auto-estimate)",
    )
    p.add_argument(
        "--no-kronos", action="store_true", help="Disable Kronos overlay in persona pipeline"
    )
    p.add_argument("--json", action="store_true", help="Output session summary as JSON")
    add_data_root_arg(p)
    add_backend_arg(p)
    return p.parse_args()


def main() -> None:  # noqa: C901
    args = _parse_args()
    symbol = normalize_symbol(args.symbol)

    cfg = ChallengeConfig(
        seed_capital=args.capital,
        target_capital=args.target,
        symbol=symbol,
    )
    tracker = ChallengeTracker(data_root=args.data_root)

    # ---- Reset --------------------------------------------------------------
    if args.reset:
        state = tracker.reset(cfg)
        print(
            f"Challenge reset.  Starting balance: ${state.balance:,.2f}  Target: ${state.target:,.2f}"
        )
        print(f"State file: {tracker.state_path()}")
        return

    # ---- Status -------------------------------------------------------------
    if args.status:
        if not tracker.exists():
            print("No active challenge.  Run `rlm challenge --reset` to start.", file=sys.stderr)
            sys.exit(1)
        _print_dashboard(tracker)
        return

    # ---- Run session --------------------------------------------------------
    if args.run:
        if not tracker.exists():
            print("No active challenge.  Run `rlm challenge --reset` first.", file=sys.stderr)
            sys.exit(1)

        directive, underlying_price, signal_alignment, confidence, iv = _get_signals(
            symbol=symbol,
            underlying_price_override=args.underlying_price,
            iv_override=args.iv,
            use_kronos=not args.no_kronos,
            data_root=args.data_root,
            backend=args.backend,
        )

        engine = ChallengeEngine(cfg, tracker)
        summary = engine.run_session(
            directive=directive,
            underlying_price=underlying_price,
            signal_alignment=signal_alignment,
            confidence=confidence,
            iv=iv,
        )

        if args.json:
            _print_summary_json(summary)
        else:
            _print_summary(summary, symbol)
        return

    # Default: show status if exists, else prompt
    if tracker.exists():
        _print_dashboard(tracker)
    else:
        print("No active challenge.  Options:")
        print("  rlm challenge --reset          Start fresh at $1,000")
        print("  rlm challenge --reset --capital 2500   Start at custom amount")


# ---------------------------------------------------------------------------
# Signal acquisition
# ---------------------------------------------------------------------------


def _get_signals(
    *,
    symbol: str,
    underlying_price_override: float | None,
    iv_override: float | None,
    use_kronos: bool,
    data_root: str | None,
    backend: str,
) -> tuple[str, float, float, float, float]:
    """Return (directive, underlying_price, signal_alignment, confidence, iv).

    Attempts to run FullRLMPipeline + PersonaDecisionPipeline.
    Falls back to neutral defaults when data is unavailable.
    """
    default_iv = iv_override or 0.18

    try:
        from rlm.core.config import build_pipeline_config
        from rlm.core.pipeline import FullRLMPipeline
        from rlm.data.readers import load_bars, load_option_chain
        from rlm.persona.pipeline import PersonaDecisionPipeline

        bars_df = load_bars(symbol, data_root=data_root, backend=backend)
        chain_df = load_option_chain(symbol, data_root=data_root, backend=backend)
        cfg = build_pipeline_config(
            symbol=symbol,
            overrides={"use_kronos": use_kronos, "attach_vix": True},
        )
        result = FullRLMPipeline(cfg).run(bars_df, chain_df)
        persona = PersonaDecisionPipeline().run(result)

        directive = persona.sisko.directive
        alignment = persona.seven.signal_alignment
        conf = persona.seven.confidence

        # Underlying price: last close from bars
        price = underlying_price_override
        if price is None and not result.factors_df.empty:
            last = result.factors_df.iloc[-1]
            price = float(last.get("close", last.get("Close", 0)) or 0)
        if not price:
            price = underlying_price_override or 500.0

        # IV: try to pull from pipeline, else fall back
        iv = default_iv
        if not result.factors_df.empty and "realized_vol" in result.factors_df.columns:
            rv = result.factors_df["realized_vol"].iloc[-1]
            if rv and rv > 0:
                iv = float(rv) * 1.1  # slight IV premium over RV

        return directive, price, alignment, conf, iv

    except Exception as exc:
        print(f"[challenge] Pipeline unavailable ({exc}); using neutral fallback.", file=sys.stderr)
        price = underlying_price_override or 500.0
        return "no_trade", price, 0.5, 0.5, default_iv


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_dashboard(tracker: ChallengeTracker) -> None:
    state = tracker.load()
    m = state.current_milestone

    bar_filled = int(state.progress_pct * 30)
    bar = "█" * bar_filled + "░" * (30 - bar_filled)

    print("\n  $1K→$25K OPTIONS CHALLENGE  ")
    print("=" * 46)
    print(f"  Balance   : ${state.balance:>12,.2f}")
    print(f"  Seed      : ${state.seed:>12,.2f}")
    print(f"  Target    : ${state.target:>12,.2f}")
    print(f"  Return    : {state.total_return_pct:>+.1f}%")
    print(f"  Progress  : [{bar}] {state.progress_pct * 100:.1f}%")
    print(f"  Stage     : {getattr(m, 'label', '—')}")
    print()
    print(f"  Sessions  : {state.session_count}")
    print(
        f"  Trades    : {len(state.trade_history)}  (W:{state.wins} L:{state.losses}  WR:{state.win_rate:.0%})"
    )
    print()

    # Milestones
    print("  Milestones:")
    for ms in MILESTONES:
        tick = "✓" if state.balance >= ms.target else " "
        print(f"    [{tick}] ${ms.target:>8,.0f}  {ms.label}")
    print()

    # Open positions
    if state.open_positions:
        print("  Open Positions:")
        for p in state.open_positions:
            pnl_sign = "+" if p.unrealised_pnl >= 0 else ""
            print(
                f"    {p.option_type.upper():4s} ${p.strike:.0f}  "
                f"×{p.qty}  DTE:{p.dte_remaining}  "
                f"P&L {pnl_sign}${p.unrealised_pnl:.2f}"
            )
    else:
        print("  Open Positions: none")
    print()
    print(f"  Data : {tracker.state_path()}")
    print()


def _print_summary(summary: object, symbol: str) -> None:
    from rlm.challenge.engine import SessionSummary

    s: SessionSummary = summary  # type: ignore[assignment]

    delta = s.balance_after - s.balance_before
    sign = "+" if delta >= 0 else ""
    print(f"\n  [{s.session_date}] {symbol} | Directive: {s.directive.upper()}")
    print(f"  Balance: ${s.balance_before:,.2f}  →  ${s.balance_after:,.2f}  ({sign}${delta:,.2f})")
    if s.message:
        for line in s.message.split("  "):
            if line.strip():
                print(f"    {line}")
    if s.milestone_cleared:
        print(f"\n  *** MILESTONE CLEARED: {s.milestone_cleared} ***")
    if s.challenge_complete:
        print("\n  *** CHALLENGE COMPLETE — $25,000 reached! ***")
    print()


def _print_summary_json(summary: object) -> None:
    from rlm.challenge.engine import SessionSummary

    s: SessionSummary = summary  # type: ignore[assignment]
    out = {
        "session_date": s.session_date,
        "directive": s.directive,
        "balance_before": s.balance_before,
        "balance_after": s.balance_after,
        "milestone_cleared": s.milestone_cleared,
        "challenge_complete": s.challenge_complete,
        "message": s.message,
        "closed_trades": [t.to_dict() for t in s.closed_trades],
        "new_position": s.new_position.to_dict() if s.new_position else None,
    }
    print(json.dumps(out, indent=2, default=str))
