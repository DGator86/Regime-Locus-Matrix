#!/usr/bin/env python3
"""Batch walk-forward OOS validation for :data:`~rlm.data.liquidity_universe.EXPANDED_LIQUID_UNIVERSE`.

Each successful symbol writes ``walkforward_summary_{SYM}.csv`` (and related artifacts) under
``data/processed/``.  Symbols with missing bars or runtime errors are skipped so the batch
always completes.

Also writes:

* ``walkforward_summary_universe_all_windows.csv`` — concatenated OOS window rows + ``symbol`` column
* ``walkforward_universe_latest.json`` — snapshot aggregates for dashboards / crew
* appends one JSON object per run to ``walkforward_universe_runs.jsonl`` — time series of
  cross-symbol OOS quality (track record / optimization feedback)

Install the systemd units in ``deploy/systemd/`` for daily unattended runs.

Examples::

    python scripts/run_walkforward_universe.py
    python scripts/run_walkforward_universe.py --no-kronos --no-vix
    python scripts/run_walkforward_universe.py --symbols SPY,QQQ
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlm.utils.compute_threads import apply_compute_thread_env  # noqa: E402

apply_compute_thread_env()

from rlm.cli.common import (  # noqa: E402
    add_backend_arg,
    add_data_root_arg,
    add_pipeline_args,
    add_profile_args,
    build_pipeline_config,
    normalize_symbol,
)
from rlm.core.services.backtest_service import BacktestRequest, BacktestService  # noqa: E402
from rlm.data.liquidity_universe import EXPANDED_LIQUID_UNIVERSE  # noqa: E402
from rlm.data.paths import get_data_root, get_processed_data_dir  # noqa: E402
from rlm.data.readers import load_bars, load_option_chain  # noqa: E402
from rlm.datasets.paths import walkforward_summary_filename  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--symbols",
        default=None,
        metavar="A,B,C",
        help="Override ticker list (default: full EXPANDED_LIQUID_UNIVERSE).",
    )
    add_pipeline_args(p)
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        out = [normalize_symbol(x) for x in str(args.symbols).split(",") if x.strip()]
        if not out:
            raise SystemExit("--symbols must list at least one ticker.")
        return out
    return list(EXPANDED_LIQUID_UNIVERSE)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _concat_universe_summaries(proc: Path) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    skip_names = frozenset({"walkforward_summary_universe_all_windows.csv"})
    for p in sorted(proc.glob("walkforward_summary_*.csv")):
        if p.name in skip_names:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        stem = p.stem.replace("walkforward_summary_", "", 1)
        sym = stem.upper()
        out = df.copy()
        out.insert(0, "symbol", sym)
        frames.append(out)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = _parse_args()
    symbols = _resolve_symbols(args)
    data_root = get_data_root(args.data_root)
    out_dir = get_processed_data_dir(args.data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    svc = BacktestService()
    ok: list[str] = []
    failed: list[dict[str, str]] = []
    sharpes: list[float] = []
    total_windows = 0

    print(
        f"[walkforward-universe] {len(symbols)} symbols → processed dir {out_dir}",
        flush=True,
    )

    for sym in symbols:
        try:
            bars_df = load_bars(sym, data_root=args.data_root, backend=args.backend)
            chain_df = load_option_chain(sym, data_root=args.data_root, backend=args.backend)
        except FileNotFoundError as e:
            failed.append({"symbol": sym, "error": f"missing data: {e}"})
            print(f"  [skip] {sym} — {e}", flush=True)
            continue
        except Exception as e:
            failed.append({"symbol": sym, "error": str(e)})
            print(f"  [skip] {sym} — load error: {e}", flush=True)
            continue

        cfg = build_pipeline_config(args, sym)
        req = BacktestRequest(
            symbol=sym,
            bars_df=bars_df,
            option_chain_df=chain_df,
            config=cfg,
            walkforward=True,
            write_outputs=True,
            out_dir=out_dir,
            initial_capital=100_000.0,
        )
        try:
            result = svc.run(req)
            svc.write_outputs(req, result)
        except Exception:
            failed.append({"symbol": sym, "error": traceback.format_exc()[-800:]})
            print(f"  [fail] {sym} — walkforward/backtest error", flush=True)
            traceback.print_exc()
            continue

        ok.append(sym)
        if result.walkforward_summary is not None and not result.walkforward_summary.empty:
            total_windows += len(result.walkforward_summary)
            if "sharpe" in result.walkforward_summary.columns:
                s = pd.to_numeric(result.walkforward_summary["sharpe"], errors="coerce")
                m = float(s.mean())
                if not np.isnan(m):
                    sharpes.append(m)

        wf_path = out_dir / walkforward_summary_filename(sym)
        print(f"  [ok] {sym} — {wf_path.name}", flush=True)

    combined = _concat_universe_summaries(out_dir)
    if combined is not None:
        uni_path = out_dir / "walkforward_summary_universe_all_windows.csv"
        combined.to_csv(uni_path, index=False)
        print(f"[walkforward-universe] wrote {uni_path.name} ({len(combined)} rows)", flush=True)

    cross_sharpe = float(np.mean(sharpes)) if sharpes else None
    ts = datetime.now(timezone.utc).isoformat()
    failed_compact = [
        {"symbol": f["symbol"], "error": (f.get("error") or "")[:500]} for f in failed
    ]
    snapshot = {
        "ts_utc": ts,
        "symbols_attempted": len(symbols),
        "symbols_ok": len(ok),
        "symbols_failed": len(failed),
        "cross_symbol_mean_window_sharpe": cross_sharpe,
        "total_oos_windows": total_windows,
        "ok": ok,
        "failed": failed_compact,
    }
    snap_path = out_dir / "walkforward_universe_latest.json"
    snap_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    print(f"[walkforward-universe] wrote {snap_path.name}", flush=True)

    _append_jsonl(out_dir / "walkforward_universe_runs.jsonl", snapshot)

    print(
        f"[walkforward-universe] done — ok={len(ok)} failed={len(failed)} "
        f"mean_window_sharpe={cross_sharpe}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
