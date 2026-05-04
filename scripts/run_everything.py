#!/usr/bin/env python3
"""
Run the **full stack** from repo root in order:

1. ``run_universe_options_pipeline.py`` — IBKR + factors + forecast + ROEE + Massive chain match + risk plan JSON
2. Optional: ``ibkr_paper_trade_from_plans.py``  — **paper** opening combos (Level 2 debit structures)
3. Optional: ``ibkr_equity_paper_trade.py``       — **paper** equity BUY/SELL from regime direction
4. ``monitor_active_trade_plans.py`` — Massive marks vs stops; optional ``--paper-close`` **MKT** exits

Examples::

    python scripts/run_everything.py --master
    python scripts/run_master.py
    python scripts/run_everything.py
    python scripts/run_everything.py --full-paper --interval 60 --rescan-interval 300
    python scripts/run_everything.py --paper-trade --paper-close --follow --interval 120
    python scripts/run_everything.py --pipeline-args "--top 2 --no-vix"
    python scripts/run_everything.py --master --with-equity              # options + equities book
    python scripts/run_everything.py --with-equity --equity-dry-run      # equity signals only, no IBKR orders
    python scripts/run_everything.py --master --telegram-bot             # + Telegram long-poll bot (``.env``)

**Master mode** (``--master``): continuous monitor, **60s** mark polls, **300s** (5 min) universe
rescans (only **Mon–Fri 09:00–16:00 US/Eastern** unless ``--scanner-24h``), and optional IBKR
**paper** option opens/closes (see ``--paper-trade`` / ``--paper-close``).
With ``--with-equity`` (recommended: ``scripts/run_master.py``), **options stay simulation-only**
(local marks, ``trade_log.csv``, state); **only equities** use IBKR. Override timing with
``--interval`` / ``--rescan-interval`` if needed.

``--with-equity`` runs a parallel equity paper book alongside the options book using the same
regime signals.  This requires no options permissions — plain stock BUY/SELL orders.  Positions
are written to ``equity_positions_state.json`` and logged to ``equity_trade_log.csv``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlm.utils.market_hours import is_scanner_window_open, scanner_window_label  # noqa: E402


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT))
    return int(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        default="data/processed/universe_trade_plans.json",
        help="Plans JSON path (relative to repo root unless absolute)",
    )
    ap.add_argument(
        "--pipeline-args",
        default="",
        help="Extra args for run_universe_options_pipeline.py (quoted), e.g. '--top 3 --no-vix'",
    )
    ap.add_argument(
        "--use-vp-gating",
        action="store_true",
        help="Enable volume-profile / 80%%-rule gating in downstream pipeline decisions.",
    )
    ap.add_argument(
        "--follow",
        action="store_true",
        help="After pipeline, run monitor in a loop (see --interval) instead of --once",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=argparse.SUPPRESS,
        help="Seconds between monitor polls when --follow (default: 120, or 60 with --master)",
    )
    ap.add_argument(
        "--rescan-interval",
        type=float,
        default=argparse.SUPPRESS,
        help="Re-run universe pipeline every N seconds in the background when --follow (default: 0, or 300 with --master)",
    )
    ap.add_argument(
        "--skip-monitor", action="store_true", help="Only run the universe options pipeline"
    )
    ap.add_argument(
        "--skip-pipeline", action="store_true", help="Only run monitor (plans file must exist)"
    )
    ap.add_argument(
        "--paper-trade",
        action="store_true",
        help="After pipeline, place opening LMT combos from plans (paper IBKR only)",
    )
    ap.add_argument("--paper-trade-max", type=int, default=10, help="Cap opening orders")
    ap.add_argument(
        "--paper-dry-run", action="store_true", help="Log openings only (no IBKR transmit)"
    )
    ap.add_argument(
        "--paper-close",
        action="store_true",
        help="Monitor transmits MKT closes on exit signals (paper IBKR only)",
    )
    ap.add_argument("--paper-close-dry-run", action="store_true", help="Log closes only")
    ap.add_argument(
        "--force-close-dte",
        type=float,
        default=0.0,
        help="Pass to monitor: force-close positions within N fractional days of expiry. "
        "0.0 = disabled; use 0.1 for 0DTE safety.",
    )
    ap.add_argument(
        "--full-paper",
        action="store_true",
        help="Shorthand: --paper-trade --paper-close --follow (continuous monitor + paper in/out)",
    )
    ap.add_argument(
        "--master",
        action="store_true",
        help=(
            "Timed rescans + monitor every 60s; implies --paper-trade --paper-close --follow "
            "unless combined with --with-equity (then options are dry-run only, no IBKR option closes)."
        ),
    )
    # -----------------------------------------------------------------------
    # Equity book flags
    # -----------------------------------------------------------------------
    ap.add_argument(
        "--with-equity",
        action="store_true",
        help=(
            "Equity-primary dual-book mode: run ibkr_equity_paper_trade.py with real IBKR orders "
            "for stocks, while options stay hypothetical — dry-run opens (signal-log), monitor "
            "updates marks/P&L locally, and no IBKR option combo opens or closes are sent."
        ),
    )
    ap.add_argument(
        "--equity-dry-run",
        action="store_true",
        help="Equity book: log signals without placing real IBKR orders",
    )
    ap.add_argument(
        "--equity-position-usd",
        type=float,
        default=10_000.0,
        help="Target notional USD per equity position (default: $10,000)",
    )
    ap.add_argument(
        "--equity-stop-pct",
        type=float,
        default=5.0,
        help="Equity hard stop (%% below entry, default: 5)",
    )
    ap.add_argument(
        "--equity-target-pct",
        type=float,
        default=10.0,
        help="Equity take-profit (%% above entry, default: 10)",
    )
    ap.add_argument(
        "--equity-risk-usd",
        type=float,
        default=0.0,
        help="Dollar amount to risk per equity trade (overrides --equity-position-usd)",
    )
    ap.add_argument(
        "--equity-use-account-scale",
        action="store_true",
        help="Scale equity positions based on IBKR account balance and AI confidence",
    )
    ap.add_argument(
        "--equity-max-account-pct",
        type=float,
        default=10.0,
        help="Max %% of account balance per equity position (default: 10)",
    )
    ap.add_argument(
        "--scanner-hours-et",
        action="store_true",
        help="Gate universe rescans to Mon–Fri 09:00–16:00 America/New_York; sets --follow and "
        "default 300s rescan if none given.",
    )
    ap.add_argument(
        "--scanner-24h",
        action="store_true",
        help="Disable US/Eastern scanner window (rescans run 24/7). Default: window on with --master.",
    )
    ap.add_argument(
        "--telegram-bot",
        action="store_true",
        help="Start scripts/rlm_telegram_bot.py in a separate process (reads TELEGRAM_* from .env).",
    )
    args = ap.parse_args()

    if args.master:
        args.paper_trade = True
        args.paper_close = True
        args.follow = True
    if args.full_paper:
        args.paper_trade = True
        args.paper_close = True
        args.follow = True

    # Equity-primary mode: IBKR for stocks only. Options are hypothetical (no combo orders).
    if args.with_equity:
        if not args.paper_dry_run:
            args.paper_dry_run = True
        if args.paper_close:
            args.paper_close = False
        print(
            "[info] --with-equity: options are simulation-only (dry-run opens, no IBKR closes); "
            "equities use IBKR per ibkr_equity_paper_trade.py",
            flush=True,
        )

    if not hasattr(args, "interval"):
        args.interval = 60.0 if args.master else 120.0
    if not hasattr(args, "rescan_interval"):
        args.rescan_interval = 300.0 if args.master else 0.0

    if args.scanner_hours_et:
        args.follow = True
        if float(args.rescan_interval) <= 0.0:
            args.rescan_interval = 300.0

    scanner_hours_et = (args.master or args.scanner_hours_et) and not args.scanner_24h
    if scanner_hours_et:
        print(
            "[info] Universe rescans gated to Mon–Fri 09:00–16:00 America/New_York "
            "(use --scanner-24h for continuous rescans).",
            flush=True,
        )

    py = sys.executable
    plans = args.out

    def pipeline_cmd() -> list[str]:
        cmd = [py, str(ROOT / "scripts" / "run_universe_options_pipeline.py"), "--out", plans]
        if args.pipeline_args.strip():
            cmd.extend(shlex.split(args.pipeline_args))
        if args.use_vp_gating and "--use-vp-gating" not in cmd:
            cmd.append("--use-vp-gating")
        return cmd

    def paper_cmd() -> list[str]:
        pcmd = [
            py,
            str(ROOT / "scripts" / "ibkr_paper_trade_from_plans.py"),
            "--plans",
            plans,
            "--max",
            str(args.paper_trade_max),
        ]
        if args.paper_dry_run:
            pcmd.append("--dry-run")
        return pcmd

    def equity_cmd() -> list[str]:
        ecmd = [
            py,
            str(ROOT / "scripts" / "ibkr_equity_paper_trade.py"),
            "--plans",
            plans,
            "--position-usd",
            str(args.equity_position_usd),
            "--stop-pct",
            str(args.equity_stop_pct),
            "--target-pct",
            str(args.equity_target_pct),
        ]
        if args.equity_risk_usd > 0:
            ecmd.extend(["--risk-usd", str(args.equity_risk_usd)])
        if args.equity_use_account_scale:
            ecmd.extend(["--use-account-scale", "--max-account-pct", str(args.equity_max_account_pct)])
        if args.equity_dry_run:
            ecmd.append("--dry-run")
        return ecmd

    if not args.skip_pipeline:
        rc = _run(pipeline_cmd())
        if rc != 0:
            return rc

    if args.paper_trade:
        rc = _run(paper_cmd())
        if rc != 0:
            print(
                f"[warn] paper-trade step exited with code {rc}; continuing to monitor", flush=True
            )

    if args.with_equity:
        # Run equity book in a background thread so it doesn't block the monitor
        def _run_equity() -> None:
            rc = _run(equity_cmd())
            if rc != 0:
                print(f"[warn] equity trade step exited with code {rc}", flush=True)

        et = threading.Thread(target=_run_equity, name="equity-trade", daemon=True)
        et.start()
        et.join(timeout=120)  # wait up to 2 min; if still running, let it continue

    if args.skip_monitor:
        return 0

    if args.follow and args.rescan_interval > 0 and (not args.skip_pipeline or args.paper_trade):
        rescan_every = max(30.0, float(args.rescan_interval))
        rescan_lock = threading.Lock()

        def _rescan_loop() -> None:
            while True:
                time.sleep(rescan_every)
                with rescan_lock:
                    if scanner_hours_et and not is_scanner_window_open():
                        print(
                            f"[rescan] skip — outside ET scanner window ({scanner_window_label()})",
                            flush=True,
                        )
                        continue
                    if not args.skip_pipeline:
                        print(f"[rescan] every {rescan_every:.0f}s: universe pipeline", flush=True)
                        if _run(pipeline_cmd()) != 0:
                            print("[rescan] pipeline failed (continuing)", flush=True)
                    if args.paper_trade:
                        print("[rescan] paper-trade from plans", flush=True)
                        if _run(paper_cmd()) != 0:
                            print("[rescan] paper-trade step failed (continuing)", flush=True)
                    if args.with_equity:
                        print("[rescan] equity paper trade", flush=True)
                        if _run(equity_cmd()) != 0:
                            print("[rescan] equity trade step failed (continuing)", flush=True)

        threading.Thread(target=_rescan_loop, name="universe-rescan", daemon=True).start()

    if args.telegram_bot:
        tscript = ROOT / "scripts" / "rlm_telegram_bot.py"
        print(f"+ [telegram] starting {tscript} (separate process; .env loaded inside bot)", flush=True)
        try:
            subprocess.Popen(
                [py, str(tscript)],
                cwd=str(ROOT),
                env=os.environ.copy(),
            )
        except OSError as e:
            print(f"[warn] could not start Telegram bot: {e}", flush=True)

    mcmd = [
        py,
        str(ROOT / "scripts" / "monitor_active_trade_plans.py"),
        "--plans",
        plans,
    ]
    if args.follow:
        mcmd.extend(["--interval", str(args.interval)])
    else:
        mcmd.append("--once")
    if args.paper_close:
        mcmd.append("--paper-close")
    if args.paper_close_dry_run:
        mcmd.append("--paper-close-dry-run")
    if args.force_close_dte > 0.0:
        mcmd.extend(["--force-close-dte", str(args.force_close_dte)])
    return _run(mcmd)


if __name__ == "__main__":
    raise SystemExit(main())
