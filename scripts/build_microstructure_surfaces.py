#!/usr/bin/env python3
"""
End-of-day microstructure surface builder.

Reads the day's greeks snapshots from the microstructure lake and computes:
  1. GEX surface  (gamma exposure vs. strike/expiry)
  2. IV surface   (interpolated implied volatility grid)

Run this once after market close (or add to nightly.bat).

Examples::

    python scripts/build_microstructure_surfaces.py --symbol SPY --date 2025-06-10
    python scripts/build_microstructure_surfaces.py --symbol SPY,QQQ --date today
    python scripts/build_microstructure_surfaces.py --symbol SPY --date today --snapshot-time 16:00:00
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


def _parse_symbols(s: str) -> list[str]:
    return [x.strip().upper() for x in s.replace(";", ",").split(",") if x.strip()]


def _resolve_date(s: str) -> str:
    if s.lower() == "today":
        return date.today().isoformat()
    return s


def build_surfaces_for_date(
    symbol: str,
    *,
    target_date: str,
    snapshot_time: str = "16:00:00",
    data_root: str,
    duckdb_conn: "duckdb.DuckDBPyConnection",
) -> None:
    """Build and persist GEX + IV surfaces for *symbol* on *target_date*."""
    from rlm.microstructure.calculators.gex import build_gex_surface, save_gex_surface
    from rlm.microstructure.calculators.iv_surface import build_iv_surface_from_parquet, save_iv_surface
    from rlm.microstructure.database.query import MicrostructureDB

    db = MicrostructureDB(data_path=data_root)
    db._conn = duckdb_conn

    # Find the last snapshot timestamp for the target date
    parquet_glob = f"{data_root}/options/{symbol}/greeks_snapshots/*.parquet"
    date_filter = f"CAST(timestamp AS DATE) = DATE '{target_date}'"
    try:
        snap_query = f"""
            SELECT DISTINCT timestamp
            FROM '{parquet_glob}'
            WHERE {date_filter}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        result = duckdb_conn.execute(snap_query).fetchdf()
    except Exception as exc:
        print(f"  [WARN] No snapshot data found for {symbol} on {target_date}: {exc}")
        return

    if result.empty:
        print(f"  [SKIP] No snapshot data for {symbol} on {target_date}")
        return

    timestamp = result.iloc[0]["timestamp"]
    print(f"  Using snapshot: {timestamp}")

    # ── GEX surface ──────────────────────────────────────────────────────────
    try:
        gex_df = build_gex_surface(duckdb_conn, symbol=symbol, timestamp=timestamp, data_path=data_root)
        if gex_df.empty:
            print(f"  [WARN] GEX surface is empty for {symbol} @ {timestamp}")
        else:
            save_gex_surface(gex_df, symbol=symbol, data_path=data_root)
            from rlm.microstructure.calculators.gex import aggregate_gex_profile, gex_flip_level
            profile = aggregate_gex_profile(gex_df)
            flip = gex_flip_level(gex_df)
            print(
                f"  GEX: net={profile['total_net_gex']:+.2e}  "
                f"regime={profile['dominant_regime']}  "
                f"flip={flip:.2f}" if flip else f"  GEX: net={profile['total_net_gex']:+.2e}  regime={profile['dominant_regime']}"
            )
    except Exception as exc:
        print(f"  [ERROR] GEX build failed for {symbol}: {exc}")

    # ── IV surface ───────────────────────────────────────────────────────────
    try:
        iv_df = build_iv_surface_from_parquet(
            duckdb_conn, symbol=symbol, timestamp=timestamp, data_path=data_root
        )
        if iv_df.empty:
            print(f"  [WARN] IV surface is empty for {symbol} @ {timestamp}")
        else:
            save_iv_surface(iv_df, symbol=symbol, data_path=data_root)
            from rlm.microstructure.calculators.iv_surface import query_iv_surface, skew_at_dte
            atm_iv = query_iv_surface(iv_df, moneyness=1.0, dte=30.0)
            skew = skew_at_dte(iv_df, dte=30.0)
            print(f"  IV:  ATM 30d={atm_iv:.1%}  skew 25d={skew:+.1%}")
    except Exception as exc:
        print(f"  [ERROR] IV surface build failed for {symbol}: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols", default="SPY", help="Comma-separated tickers")
    p.add_argument(
        "--date", default="today",
        help="Target date YYYY-MM-DD or 'today' (default: today)",
    )
    p.add_argument(
        "--snapshot-time", default="16:00:00",
        help="Preferred snapshot time HH:MM:SS (closest to close, default: 16:00:00)",
    )
    p.add_argument(
        "--data-root", default="data/microstructure",
        help="Root of microstructure lake (relative to repo root)",
    )
    args = p.parse_args()

    symbols = _parse_symbols(args.symbols)
    target_date = _resolve_date(args.date)
    data_root = str(ROOT / args.data_root)

    try:
        import duckdb
        conn = duckdb.connect()
    except ImportError:
        raise SystemExit(
            "DuckDB is required.\n"
            "Install: pip install 'regime-locus-matrix[microstructure]'"
        )

    print(f"\nBuilding microstructure surfaces for {', '.join(symbols)} on {target_date}\n")
    for sym in symbols:
        print(f"── {sym} ──")
        build_surfaces_for_date(
            sym,
            target_date=target_date,
            snapshot_time=args.snapshot_time,
            data_root=data_root,
            duckdb_conn=conn,
        )
        print()

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
