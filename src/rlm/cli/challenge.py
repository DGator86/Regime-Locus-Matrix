"""``rlm challenge`` — $1,000 → $25,000 PDT-aware dry-run challenge CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm challenge",
        description="Run or inspect the $1k→$25k dry-run challenge engine.",
    )
    p.add_argument(
        "--status", action="store_true",
        help="Print current challenge account state and PDT budget",
    )
    p.add_argument(
        "--run", action="store_true",
        help="Run one challenge decision cycle for --symbol",
    )
    p.add_argument("--symbol", default=None, help="Ticker (e.g. SPY)")
    p.add_argument(
        "--reset", action="store_true",
        help="Reset challenge state to starting equity ($1,000)",
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Override challenge data directory (default: data/challenge/)",
    )
    return p.parse_args()


def main() -> None:  # noqa: C901
    args = _parse_args()

    from rlm.challenge.state import ChallengeStateManager

    data_dir = Path(args.data_dir) if args.data_dir else None
    mgr = ChallengeStateManager(data_dir=data_dir)

    if args.reset:
        mgr.reset()
        print("[challenge] State reset to $1,000 starting equity.")
        return

    state, pdt = mgr.load()

    if args.status:
        pct_to_goal = (state.current_equity / 25_000.0) * 100.0
        print(f"\n{'='*55}")
        print(f"  CHALLENGE STATUS — $1k → $25k")
        print(f"{'='*55}")
        print(f"  Current equity : ${state.current_equity:>10,.2f}")
        print(f"  Peak equity    : ${state.peak_equity:>10,.2f}")
        print(f"  Progress       : {pct_to_goal:.1f}% of $25,000 goal")
        print(f"  Realized P&L   : ${state.realized_pnl:>+10,.2f}")
        print(f"  Open positions : {state.open_positions_count}")
        print(f"  Sessions run   : {state.sessions_run}")
        print(f"  W/L            : {state.wins}/{state.losses} "
              f"({state.win_rate*100:.0f}% win rate)")
        print(f"\n  PDT slots remaining (rolling 5d): {pdt.day_trades_remaining}")
        print(f"  Same-day exit allowed           : {pdt.same_day_exit_allowed}")
        print(f"  Must hold overnight if entered  : {pdt.must_hold_overnight_if_entered}")
        print(f"{'='*55}\n")
        return

    if args.run:
        if not args.symbol:
            print("[challenge] --symbol is required with --run", file=sys.stderr)
            sys.exit(1)

        # Build a synthetic persona input from available pipeline artifacts
        # (In production, wire to actual pipeline output; here we show the hook)
        from rlm.challenge.pipeline import ChallengeDecisionPipeline
        from rlm.persona.models import PersonaPipelineInput
        from rlm.persona.pipeline import PersonaDecisionPipeline

        sym = args.symbol.upper()
        print(f"[challenge] Running decision cycle for {sym} …")

        # Try to load pipeline artifacts from data/processed/ if they exist
        persona_inp = _build_persona_input(sym)
        persona_result = PersonaDecisionPipeline().run(persona_inp)

        directive = ChallengeDecisionPipeline().run(sym, persona_result, state, pdt)

        print(json.dumps({
            "symbol": directive.symbol,
            "directive": directive.directive,
            "trade_mode": directive.trade_mode,
            "conviction": directive.conviction,
            "setup_score": directive.setup_score,
            "pdt_slots_remaining": directive.pdt_slots_remaining,
            "same_day_exit_allowed": directive.same_day_exit_allowed,
            "contract_profile": {
                "delta_band": [directive.contract_profile.target_delta_min,
                               directive.contract_profile.target_delta_max],
                "dte_band": [directive.contract_profile.preferred_dte_min,
                             directive.contract_profile.preferred_dte_max],
                "max_spread_pct": directive.contract_profile.max_spread_pct,
                "note": directive.contract_profile.note,
            },
            "risk_plan": {
                "premium_outlay_pct": directive.risk_plan.premium_outlay_pct,
                "hard_stop_pct": directive.risk_plan.hard_stop_pct,
                "trail_activate_pct": directive.risk_plan.trail_activate_pct,
                "profit_target_pct": directive.risk_plan.profit_target_pct,
                "partial_take_pct": directive.risk_plan.partial_take_pct,
            },
            "reason_summary": directive.reason_summary,
            "persona": {
                "seven": {"bias": persona_result.seven.bias,
                          "confidence": persona_result.seven.confidence},
                "garak": {"trap_risk": persona_result.garak.trap_risk,
                          "veto": persona_result.garak.veto},
                "sisko": {"directive": persona_result.sisko.directive},
                "data": {"regime_match": persona_result.data.regime_match,
                         "historical_edge": persona_result.data.historical_edge},
            },
        }, indent=2))
        return

    # Default: print help
    from rlm.cli.challenge import _parse_args as _p
    _p().print_help()


def _build_persona_input(symbol: str):
    """Try to read existing pipeline artifacts, otherwise use neutral defaults."""
    from rlm.persona.models import PersonaPipelineInput
    import os

    plans_path = Path("data") / "processed" / "universe_trade_plans.json"
    if plans_path.exists():
        try:
            with open(plans_path) as fh:
                plans = json.load(fh)
            for plan in plans:
                if plan.get("symbol", "").upper() == symbol:
                    meta = plan.get("decision", {}).get("metadata", {})
                    return PersonaPipelineInput(
                        symbol=symbol,
                        regime_label=str(meta.get("regime_label", "unknown")),
                        regime_confidence=float(meta.get("regime_confidence", 0.5)),
                        forecast_return=float(meta.get("forecast_return", 0.0)),
                        signal_alignment=float(meta.get("signal_alignment", 0.5)),
                        historical_edge=float(meta.get("historical_edge", 0.5)),
                    )
        except Exception:
            pass

    # Fallback: neutral defaults
    return PersonaPipelineInput(symbol=symbol)
