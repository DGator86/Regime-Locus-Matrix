"""Ping external data paths used by RLM.

**Massive** is used for **options only** (REST + OPRA flat-file S3). **Equity** bars are **IBKR**
(plus optional yfinance / stooq checks here).

Run from repo root (loads ``.env`` via dotenv inside clients):

    python scripts/super_ping_data.py
    python scripts/super_ping_data.py --ticker QQQ
    python scripts/super_ping_data.py --strict-ibkr   # exit 2 if IBKR fails

Exit codes:
    0 — Massive options checks OK (and ``--strict-ibkr`` satisfied if set)
    1 — Missing ``MASSIVE_API_KEY`` or Massive options checks failed
    2 — ``--strict-ibkr`` set and IBKR failed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _line(status: str, name: str, detail: str) -> None:
    print(f"[{status:12}] {name}: {detail}")


def _massive_http_code(exc: BaseException) -> int | None:
    s = str(exc)
    if "Massive HTTP " in s:
        try:
            return int(s.split("Massive HTTP ")[1].split(" ")[0])
        except (IndexError, ValueError):
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ping RLM data inputs")
    parser.add_argument("--ticker", default="SPY", help="Underlying for Massive / yfinance checks")
    parser.add_argument(
        "--strict-ibkr",
        action="store_true",
        help="Exit with code 2 if IBKR historical bars fail",
    )
    args = parser.parse_args()
    sym = str(args.ticker).upper()

    failures = 0
    ibkr_failed = False

    # --- Massive ---
    try:
        from rlm.data.massive import MassiveClient
    except ImportError as e:
        _line("FAIL", "Massive import", str(e))
        raise SystemExit(1)

    try:
        client = MassiveClient()
    except ValueError as e:
        _line("FAIL", "Massive API key", str(e))
        raise SystemExit(1)

    def try_massive(label: str, fn: Callable[[], Any], *, entitlement_ok: bool = False) -> None:
        nonlocal failures
        try:
            data = fn()
            if isinstance(data, dict) and data.get("status") == "OK":
                n = len(data.get("results") or [])
                _line("PASS", label, f"status=OK, results={n}")
            elif isinstance(data, dict) and "results" in data:
                n = len(data.get("results") or [])
                _line("PASS", label, f"results={n}")
            elif isinstance(data, dict):
                _line("PASS", label, f"keys={list(data.keys())[:8]}")
            else:
                _line("PASS", label, repr(data)[:120])
        except Exception as e:
            code = _massive_http_code(e)
            if entitlement_ok and code == 403:
                _line(
                    "ENTITLEMENT",
                    label,
                    "HTTP 403 (Massive plan); options-only usage — see Massive dashboard / support",
                )
            else:
                failures += 1
                _line("FAIL", label, str(e)[:500])

    _line("INFO", "Massive base", client.base_url)

    try_massive(
        f"Massive options snapshot {sym}",
        lambda: client.option_chain_snapshot(sym, limit=2),
        entitlement_ok=False,
    )

    try_massive(
        f"Massive options contracts ref ({sym})",
        lambda: client.option_contracts_reference(underlying_ticker=sym, limit=3),
        entitlement_ok=True,
    )

    # Option chain helpers (in-process)
    try:
        from rlm.data.massive_option_chain import massive_option_snapshot_to_normalized_chain
        from rlm.data.occ_symbol import parse_occ_option_symbol

        snap = client.option_chain_snapshot(sym, limit=1)
        if isinstance(snap, dict) and snap.get("results"):
            df = massive_option_snapshot_to_normalized_chain(snap, underlying=sym)
            raw0 = snap["results"][0]
            occ = str(raw0.get("details", {}).get("ticker") or "")
            parsed = parse_occ_option_symbol(occ) if occ else None
            _line(
                "PASS",
                "RLM option normalize + OCC",
                f"rows={len(df)} occ={occ[:36]!r} strike={parsed.strike if parsed else None}",
            )
        else:
            _line("WARN", "RLM option normalize + OCC", "no results to parse")
    except Exception as e:
        failures += 1
        _line("FAIL", "RLM option normalize + OCC", str(e)[:300])

    # --- IBKR ---
    try:
        from rlm.data.ibkr_stocks import fetch_historical_stock_bars

        df = fetch_historical_stock_bars(sym, duration="5 D", bar_size="1 day", timeout_sec=60.0)
        if df.empty:
            _line("WARN", "IBKR historical bars", f"{sym} returned 0 rows")
            ibkr_failed = True
        else:
            _line(
                "PASS",
                "IBKR historical bars",
                f"{sym} rows={len(df)} last_close={df['close'].iloc[-1]} vwap_ok={not df['vwap'].isna().all()}",
            )
    except ImportError as e:
        _line("SKIP", "IBKR historical bars", f"ibapi missing: {e}")
        ibkr_failed = True
    except Exception as e:
        _line("FAIL", "IBKR historical bars", str(e)[:400])
        ibkr_failed = True

    # --- Massive flat files (S3; separate from REST key) ---
    try:
        from botocore.exceptions import ClientError

        from rlm.data.massive_flatfiles import boto3_s3_client, load_massive_flatfiles_config

        cfg = load_massive_flatfiles_config()
        s3 = boto3_s3_client(cfg)
        prefix = "us_options_opra/trades_v1/"
        lst = s3.list_objects_v2(Bucket=cfg.bucket, Prefix=prefix, MaxKeys=2)
        n = int(lst.get("KeyCount") or 0)
        _line("PASS", "Massive S3 list_objects", f"bucket={cfg.bucket!r} prefix={prefix!r} keys={n}")
        contents = lst.get("Contents") or []
        if contents:
            key0 = contents[0]["Key"]
            try:
                obj = s3.get_object(Bucket=cfg.bucket, Key=key0)
                obj["Body"].read(1)
                _line("PASS", "Massive S3 get_object", f"read 1 byte from {key0[:56]}…")
            except ClientError as e:
                err = e.response.get("Error", {})
                if err.get("Code") in ("403", "Forbidden"):
                    _line(
                        "ENTITLEMENT",
                        "Massive S3 get_object",
                        "HTTP 403 — list works but download blocked; contact Massive for GetObject on flat files",
                    )
                else:
                    failures += 1
                    _line("FAIL", "Massive S3 get_object", str(err)[:300])
        else:
            _line("WARN", "Massive S3 get_object", "no keys under prefix to probe")
    except ValueError as e:
        _line("SKIP", "Massive S3 flat files", str(e)[:200])
    except ImportError as e:
        _line("SKIP", "Massive S3 flat files", f"boto3 missing: {e}")
    except Exception as e:
        failures += 1
        _line("FAIL", "Massive S3 flat files", str(e)[:400])

    # --- Optional: yfinance / pandas-datareader (declared in pyproject; may be unused by pipelines) ---
    try:
        import yfinance as yf

        h = yf.Ticker(sym).history(period="5d", auto_adjust=False)
        _line("PASS", "yfinance history", f"{sym} rows={len(h)} cols={list(h.columns)[:5]}")
    except Exception as e:
        _line("SKIP", "yfinance history", str(e)[:200])

    try:
        import pandas_datareader.data as web

        # Stooq is unauthenticated; symbol format for US stocks
        stooq_sym = sym.lower() + ".us"
        h = web.DataReader(stooq_sym, "stooq")
        n = len(h)
        if n == 0:
            _line(
                "WARN",
                "pandas_datareader (stooq)",
                f"{stooq_sym} rows=0 (often rate/format flakiness; not a core RLM path)",
            )
        else:
            _line("PASS", "pandas_datareader (stooq)", f"{stooq_sym} rows={n}")
    except Exception as e:
        _line("SKIP", "pandas_datareader (stooq)", str(e)[:200])

    print("---")
    if failures:
        print(f"Summary: {failures} hard failure(s) on Massive options / RLM helpers.")
        raise SystemExit(1)
    if args.strict_ibkr and ibkr_failed:
        print("Summary: --strict-ibkr and IBKR did not succeed.")
        raise SystemExit(2)
    print(
        "Summary: Massive options (REST + OPRA S3 list/get probe) + RLM helpers OK; "
        "equity via IBKR. See ENTITLEMENT/SKIP for Massive plan or optional probes."
    )


if __name__ == "__main__":
    main()
