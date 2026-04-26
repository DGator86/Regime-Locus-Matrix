"""``rlm activate`` — unified RLM stack activation.

Triggers three plans simultaneously after a bulk data import:

  1. **Equity plan**   — IBKR paper equity BUY/SELL from regime signals
  2. **Options plan**  — universe options pipeline + marks monitor (continuous)
  3. **Challenge plan** — $1K→$25K aggressive session for SPY + QQQ

Phase 1 (ingest) and Phase 2 (activate) are separated so that all three
downstream plans share freshly-imported data.

Usage::

    rlm activate
    rlm activate --no-ingest          # use cached data, skip Phase 1
    rlm activate --no-challenge       # skip challenge sessions
    rlm activate --equity-dry-run     # log equity signals, no IBKR orders
    rlm activate --ingest-workers 6   # faster parallel import
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rlm.challenge.models import ChallengePipelineConfig

ROOT = Path(__file__).resolve().parents[3]   # repo root: .../Regime-Locus-Matrix
_SCRIPTS = ROOT / "scripts"

# Challenge symbols sourced from the pipeline config so they stay in sync.
CHALLENGE_SYMBOLS: tuple[str, ...] = tuple(ChallengePipelineConfig().allowed_universe)


def _ingest_symbol(symbol: str, data_root: str | None) -> tuple[str, bool, str]:
    """Ingest one symbol; returns (symbol, ok, message)."""
    cmd = [sys.executable, "-m", "rlm", "ingest", "--symbol", symbol]
    if data_root:
        cmd += ["--data-root", data_root]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    ok = r.returncode == 0
    return symbol, ok, (r.stdout.strip() or r.stderr.strip())


def phase1_ingest(
    symbols: tuple[str, ...],
    data_root: str | None,
    workers: int,
) -> None:
    print(f"\n[PHASE 1] Bulk import — {len(symbols)} symbols, {workers} parallel workers")
    print("=" * 60)
    t0 = time.monotonic()
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_ingest_symbol, s, data_root): s for s in symbols}
        for fut in as_completed(futs):
            sym, ok, msg = fut.result()
            tag = "OK  " if ok else "FAIL"
            print(f"  [{tag}] {sym:6s}  {msg[:80] if msg else ''}", flush=True)
            if not ok:
                failed.append(sym)
    elapsed = time.monotonic() - t0
    ok_count = len(symbols) - len(failed)
    print(f"\n  Ingest complete in {elapsed:.1f}s — {ok_count}/{len(symbols)} succeeded.")
    if failed:
        print(f"  Symbols that failed: {', '.join(failed)}")
    print()


def phase2_activate(
    *,
    equity_dry_run: bool,
    pipeline_args: str,
    no_challenge: bool,
    data_root: str | None,
) -> None:
    print("[PHASE 2] Activating all plans")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Equity + Options stack (long-running background process)           #
    # ------------------------------------------------------------------ #
    stack_cmd = [
        sys.executable,
        str(_SCRIPTS / "run_everything.py"),
        "--master",
        "--with-equity",
    ]
    if equity_dry_run:
        stack_cmd.append("--equity-dry-run")
    if pipeline_args.strip():
        stack_cmd += ["--pipeline-args", pipeline_args]

    print(f"  [EQUITY+OPTIONS] Starting: {' '.join(stack_cmd)}", flush=True)
    stack_proc = subprocess.Popen(stack_cmd, cwd=str(ROOT))
    print(f"  [EQUITY+OPTIONS] PID {stack_proc.pid} — running in background\n")

    # ------------------------------------------------------------------ #
    #  Challenge sessions (one per symbol, sequential)                    #
    # ------------------------------------------------------------------ #
    if not no_challenge:
        print(f"  [CHALLENGE] Running sessions for: {', '.join(CHALLENGE_SYMBOLS)}\n")
        for sym in CHALLENGE_SYMBOLS:
            ch_cmd = [sys.executable, "-m", "rlm", "challenge", "--run", "--symbol", sym]
            if data_root:
                ch_cmd += ["--data-root", data_root]
            print(f"  [CHALLENGE] {sym}", flush=True)
            subprocess.run(ch_cmd, cwd=str(ROOT))
        print()
    else:
        print("  [CHALLENGE] Skipped (--no-challenge)\n")

    # ------------------------------------------------------------------ #
    #  Hand off — keep the stack running in the foreground               #
    # ------------------------------------------------------------------ #
    print(
        f"[PHASE 2] Challenge sessions complete.  "
        f"Equity+options stack (PID {stack_proc.pid}) is running.\n"
        f"  Press Ctrl+C to shut down.\n",
        flush=True,
    )
    try:
        stack_proc.wait()
    except KeyboardInterrupt:
        print("\n[activate] Interrupt — stopping equity+options stack...", flush=True)
        stack_proc.terminate()
        try:
            stack_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            stack_proc.kill()
        print("[activate] Shutdown complete.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm activate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--no-ingest", action="store_true",
        help="Skip Phase 1 bulk import and use cached data",
    )
    p.add_argument(
        "--no-challenge", action="store_true",
        help="Skip challenge sessions in Phase 2",
    )
    p.add_argument(
        "--equity-dry-run", action="store_true",
        help="Equity plan: log signals only, no IBKR orders",
    )
    p.add_argument(
        "--pipeline-args", default="",
        help="Extra args forwarded to run_universe_options_pipeline.py (quoted)",
    )
    p.add_argument(
        "--ingest-workers", type=int, default=4,
        help="Parallel workers for bulk import (default: 4)",
    )
    p.add_argument(
        "--data-root", default=None,
        help="Override default data root directory",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Full symbol set: main universe + challenge symbols (deduped)
    from rlm.data.liquidity_universe import EXPANDED_LIQUID_UNIVERSE
    all_symbols: tuple[str, ...] = tuple(
        dict.fromkeys(EXPANDED_LIQUID_UNIVERSE + CHALLENGE_SYMBOLS)
    )

    if not args.no_ingest:
        phase1_ingest(all_symbols, data_root=args.data_root, workers=args.ingest_workers)
    else:
        print("[PHASE 1] Skipped (--no-ingest).\n")

    phase2_activate(
        equity_dry_run=args.equity_dry_run,
        pipeline_args=args.pipeline_args,
        no_challenge=args.no_challenge,
        data_root=args.data_root,
    )
