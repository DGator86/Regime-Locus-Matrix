#!/usr/bin/env python3
"""
**Universe → full equity pipeline → ROEE → Massive option chain → match legs → risk plan.**

For each symbol (default: ``LIQUID_UNIVERSE`` = Mag7 + SPY + QQQ):

1. IBKR daily bars → ``prepare_bars_for_factors`` → factors → state matrix → forecast (latest bar).
2. ``select_trade_for_row`` → strategy + abstract legs.
3. If **enter**: fetch **full** Massive option snapshot (paginated), normalize, DTE slice, ``match_legs_to_chain``.
4. Build **entry debit** (bid/ask), **mid mark** ``V0``, take-profit / hard-stop / trailing activation levels
   (:mod:`rlm.execution.risk_targets`), plus JSON suitable for ``ibkr_place_roee_combo.py``.

Output: JSON file consumed by ``scripts/monitor_active_trade_plans.py``.

Examples::

    python scripts/run_universe_options_pipeline.py --out data/processed/universe_trade_plans.json
    python scripts/run_universe_options_pipeline.py --top 3 --stop-loss-frac 0.45
    python scripts/run_universe_options_pipeline.py --workers 4

Use ``--workers`` > 1 to process symbols concurrently (IBKR history is serialized; Massive snapshot
calls run in parallel). Watch Massive rate limits.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.data.event_calendar import has_major_event_today
from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.data.liquidity_universe import LIQUID_UNIVERSE
from rlm.data.massive import MassiveClient
from rlm.data.massive_option_chain import massive_option_chain_from_client
from rlm.data.option_chain import select_nearest_expiry_slice
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.execution.risk_targets import build_spread_exit_thresholds
from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.pipeline import ForecastPipeline
from rlm.roee.chain_match import estimate_entry_cost_from_matched_legs, estimate_mark_value_from_matched_legs, match_legs_to_chain
from rlm.roee.decision import select_trade_for_row
from rlm.scoring.state_matrix import classify_state_matrix
from rlm.utils.market_hours import entry_window_open, session_label

# IBKR historical requests use one socket per call; same client_id cannot safely overlap.
_IBKR_HIST_LOCK = threading.Lock()


def _parse_symbols(s: str) -> list[str]:
    parts = [x.strip().upper() for x in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _filter_tradable_chain(chain: pd.DataFrame) -> pd.DataFrame:
    if chain.empty:
        return chain
    b = pd.to_numeric(chain["bid"], errors="coerce")
    a = pd.to_numeric(chain["ask"], errors="coerce")
    ok = np.isfinite(b) & np.isfinite(a) & (a >= b) & (b >= 0)
    return chain.loc[ok].copy()


def _one_symbol(
    sym: str,
    *,
    duration: str,
    bar_size: str,
    move_window: int,
    vol_window: int,
    strike_increment: float,
    attach_vix: bool,
    massive_limit: int,
    stop_loss_frac: float,
    trail_activate_frac: float,
    trail_retrace_frac: float,
    client: MassiveClient,
    serialize_ibkr: bool,
    market_hours_only: bool,
    buffer_open_minutes: int,
    buffer_close_minutes: int,
    event_lookahead_days: int,
    short_dte: bool,
    dte_min_override: int | None,
    dte_max_override: int | None,
) -> dict[str, object]:
    run_at = datetime.now(timezone.utc).isoformat()
    base: dict[str, object] = {
        "symbol": sym,
        "run_at_utc": run_at,
        "status": "skipped",
    }

    if market_hours_only and not entry_window_open(
        buffer_open_minutes=buffer_open_minutes,
        buffer_close_minutes=buffer_close_minutes,
    ):
        base["skip_reason"] = f"outside_entry_window ({session_label()})"
        return base

    if serialize_ibkr:
        with _IBKR_HIST_LOCK:
            bars = fetch_historical_stock_bars(
                sym,
                duration=duration,
                bar_size=bar_size,
                timeout_sec=120.0,
            )
    else:
        bars = fetch_historical_stock_bars(
            sym,
            duration=duration,
            bar_size=bar_size,
            timeout_sec=120.0,
        )
    if bars.empty:
        base["skip_reason"] = "no_ibkr_bars"
        return base

    df = bars.sort_values("timestamp").set_index("timestamp")
    df = prepare_bars_for_factors(df, option_chain=None, underlying=sym, attach_vix=attach_vix)

    feats = FactorPipeline().run(df)
    feats = classify_state_matrix(feats)
    forecast = ForecastPipeline(move_window=move_window, vol_window=vol_window)
    out = forecast.run(feats)
    out = out.copy()
    _event_flag = has_major_event_today(sym, lookahead_days=event_lookahead_days)
    out["has_major_event"] = _event_flag

    last = out.iloc[-1]
    ts = last.name if isinstance(last.name, pd.Timestamp) else pd.Timestamp.utcnow().tz_localize(None)

    pipeline_row = {
        "close": float(last["close"]),
        "sigma": float(last["sigma"]) if pd.notna(last["sigma"]) else None,
        "mean_price": float(last["mean_price"]) if pd.notna(last.get("mean_price")) else None,
        "lower_1s": float(last["lower_1s"]) if pd.notna(last.get("lower_1s")) else None,
        "upper_1s": float(last["upper_1s"]) if pd.notna(last.get("upper_1s")) else None,
        "lower_2s": float(last["lower_2s"]) if pd.notna(last.get("lower_2s")) else None,
        "upper_2s": float(last["upper_2s"]) if pd.notna(last.get("upper_2s")) else None,
        "regime_key": str(last.get("regime_key", "")),
        "S_D": float(last["S_D"]) if pd.notna(last["S_D"]) else None,
        "S_V": float(last["S_V"]) if pd.notna(last["S_V"]) else None,
        "S_L": float(last["S_L"]) if pd.notna(last["S_L"]) else None,
        "S_G": float(last["S_G"]) if pd.notna(last["S_G"]) else None,
    }
    base["pipeline"] = pipeline_row

    decision = select_trade_for_row(last, strike_increment=strike_increment, short_dte=short_dte)
    base["decision"] = {
        "action": decision.action,
        "strategy_name": decision.strategy_name,
        "rationale": decision.rationale,
        "size_fraction": decision.size_fraction,
        "regime_key": decision.regime_key,
        "metadata": {k: v for k, v in (decision.metadata or {}).items() if k != "matched_legs"},
    }

    if decision.action != "enter" or not decision.candidate or not decision.legs:
        base["skip_reason"] = "roee_skip_or_no_legs"
        return base

    try:
        chain = massive_option_chain_from_client(
            client,
            sym,
            timestamp=ts,
            limit=int(massive_limit),
        )
    except Exception as e:
        base["skip_reason"] = f"massive_chain_error:{e}"
        return base

    chain = _filter_tradable_chain(chain)
    if chain.empty:
        base["skip_reason"] = "no_tradable_option_quotes"
        return base

    c = decision.candidate
    dte_lo = dte_min_override if dte_min_override is not None else c.target_dte_min
    dte_hi = dte_max_override if dte_max_override is not None else c.target_dte_max
    expiry_slice = select_nearest_expiry_slice(chain, dte_lo, dte_hi)
    if expiry_slice.empty:
        base["skip_reason"] = "no_contracts_in_dte_window"
        base["dte_window"] = [dte_lo, dte_hi]
        return base

    matched = match_legs_to_chain(decision=decision, chain_slice=expiry_slice)
    if matched.action != "enter" or not matched.metadata.get("matched_legs"):
        base["skip_reason"] = matched.rationale or "chain_match_failed"
        return base

    mlegs = matched.metadata["matched_legs"]
    entry_debit = estimate_entry_cost_from_matched_legs(matched)
    v0 = estimate_mark_value_from_matched_legs(mlegs)

    if not np.isfinite(entry_debit) or abs(float(entry_debit)) < 1e-6:
        base["skip_reason"] = "invalid_entry_debit"
        return base
    if not np.isfinite(v0):
        base["skip_reason"] = "invalid_v0_mark"
        return base

    tp_pct = float(c.target_profit_pct)
    debit_mag = abs(float(entry_debit))
    thresholds = build_spread_exit_thresholds(
        v0=float(v0),
        entry_debit=debit_mag,
        target_profit_pct=tp_pct,
        stop_loss_frac_of_debit=stop_loss_frac,
        trail_activate_frac_of_debit=trail_activate_frac,
        trail_retrace_frac_from_peak=trail_retrace_frac,
    )

    sn = str(decision.strategy_name or "").lower()
    # All strategies are now Level 2 debit structures — the parent BAG order always BUYs.
    # Credit only if debit is unexpectedly negative (fallback guard).
    credit_style = float(entry_debit) < 0
    combo_order_action = "SELL" if credit_style else "BUY"
    # IBKR BAG limit price is dollars-per-share of the underlying (same quoting convention
    # as a single-leg option price).  estimate_entry_cost already applied × 100 to get
    # per-contract dollars, so we divide back to get the per-share IBKR price.
    lim = float(round(debit_mag / 100.0, 4))

    ibkr_spec = {
        "underlying": sym,
        "quantity": 1,
        "limit_price": lim,
        "combo_order_action": combo_order_action,
        "legs": [
            {
                "side": str(m["side"]),
                "option_type": str(m["option_type"]),
                "strike": float(m["strike"]),
                "expiry": str(m["expiry"]),
            }
            for m in mlegs
        ],
    }

    conf = float((decision.metadata or {}).get("confidence") or 0.0)
    sf = float(decision.size_fraction or 0.0)
    score = conf * sf

    base.update(
        {
            "status": "active",
            "plan_id": f"{sym}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M')}",
            "skip_reason": "",
            "matched_legs": mlegs,
            "entry_debit_dollars": float(entry_debit),
            "entry_mid_mark_dollars": float(v0),
            "thresholds": {
                "v_take_profit": thresholds.v_take_profit,
                "v_hard_stop": thresholds.v_hard_stop,
                "v_trail_activate": thresholds.v_trail_activate,
                "trail_retrace_frac": thresholds.trail_retrace_frac,
            },
            "regime_key": str(decision.regime_key or ""),
            "regime_direction": str((decision.regime_key or "").split("|")[0]),
            "candidate": {
                "target_dte_min": c.target_dte_min,
                "target_dte_max": c.target_dte_max,
                "target_profit_pct": c.target_profit_pct,
                "max_risk_pct": c.max_risk_pct,
            },
            "ibkr_combo_spec": ibkr_spec,
            "rank_score": score,
        }
    )
    return base


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default=",".join(LIQUID_UNIVERSE), help="Comma-separated tickers")
    p.add_argument("--duration", default="180 D", help="IBKR history window (e.g. '180 D', '5 D')")
    p.add_argument(
        "--bar-size",
        default="1 day",
        help="IBKR bar size string (e.g. '1 day', '15 mins', '5 mins'). "
        "Use '5 D' for --duration when fetching 5m intraday bars.",
    )
    p.add_argument("--move-window", type=int, default=100,
                   help="Forecast rolling window in bars (use 390 for 5m intraday bars)")
    p.add_argument("--vol-window", type=int, default=100)
    p.add_argument(
        "--market-hours-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip symbols outside the RTH entry window (default: True). Use --no-market-hours-only to disable.",
    )
    p.add_argument("--buffer-open-minutes", type=int, default=15,
                   help="Ignore the first N minutes after open (default 15)")
    p.add_argument("--buffer-close-minutes", type=int, default=30,
                   help="Ignore the last N minutes before close (default 30)")
    p.add_argument(
        "--event-lookahead-days",
        type=int,
        default=1,
        help="Flag symbols with earnings or macro events within N calendar days (default 1)",
    )
    p.add_argument("--strike-increment", type=float, default=1.0)
    p.add_argument(
        "--short-dte",
        action="store_true",
        default=False,
        help="Select 0DTE / 1DTE intraday strategies instead of the default 14–60 DTE structures.",
    )
    p.add_argument(
        "--dte-min",
        type=int,
        default=None,
        help="Override the strategy's target_dte_min when matching the option chain slice.",
    )
    p.add_argument(
        "--dte-max",
        type=int,
        default=None,
        help="Override the strategy's target_dte_max when matching the option chain slice.",
    )
    p.add_argument("--ibkr-delay", type=float, default=0.35, help="Pause between symbols (workers==1 only)")
    p.add_argument("--massive-delay", type=float, default=0.35, help="Pause after each Massive snapshot pull (workers==1 only)")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent symbols (default 1 sequential). IBKR bars are locked; Massive runs in parallel.",
    )
    p.add_argument("--no-vix", action="store_true")
    p.add_argument("--massive-limit", type=int, default=250, help="Per-page limit for option snapshot pagination")
    p.add_argument("--stop-loss-frac", type=float, default=0.5, help="Hard stop = V0 - frac * entry_debit")
    p.add_argument("--trail-activate-frac", type=float, default=0.15, help="Start trailing after V0 + frac * D")
    p.add_argument("--trail-retrace-frac", type=float, default=0.25, help="Trail stop = peak * (1 - this)")
    p.add_argument("--top", type=int, default=0, help="If >0, only keep N highest rank_score among active plans")
    p.add_argument(
        "--allow-duplicates",
        action="store_true",
        default=False,
        help="Allow new active plans even when the symbol already has an open position in monitor state",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/universe_trade_plans.json"),
        help="Output JSON (under repo root if relative)",
    )
    args = p.parse_args()

    # Auto-scale defaults for intraday bar sizes so the user can simply pass
    # --bar-size "5 mins" without having to also override --duration and --move-window.
    _intraday_bar_sizes = {"5 mins", "5 min", "5m", "15 mins", "15 min", "15m", "1 min", "1 hour"}
    _bar_size_norm = args.bar_size.strip().lower()
    _is_intraday = any(_bar_size_norm.startswith(x.lower()) for x in _intraday_bar_sizes) or (
        "min" in _bar_size_norm or "hour" in _bar_size_norm
    )
    if _is_intraday:
        # If the user kept the daily default for --duration, switch to 5 D (≈5 RTH sessions)
        if args.duration == "180 D":
            args.duration = "5 D"
            print(f"[intraday] --duration auto-set to '{args.duration}' for bar-size '{args.bar_size}'")
        # If the user kept the daily default for --move-window, switch to 390 (1 RTH week at 5m)
        if args.move_window == 100:
            args.move_window = 390
            args.vol_window = 390
            print(f"[intraday] --move-window / --vol-window auto-set to 390 for bar-size '{args.bar_size}'")

    syms = _parse_symbols(args.symbols)
    results: list[dict[str, object]] = []
    workers = max(1, int(args.workers))

    try:
        shared_massive = MassiveClient() if workers == 1 else None
    except ValueError as e:
        print(f"Massive client: {e}", file=sys.stderr)
        return 1

    def _run_sym(sym: str) -> dict[str, object]:
        client = shared_massive if shared_massive is not None else MassiveClient()
        try:
            return _one_symbol(
                sym,
                duration=args.duration,
                bar_size=args.bar_size,
                move_window=args.move_window,
                vol_window=args.vol_window,
                strike_increment=args.strike_increment,
                attach_vix=not args.no_vix,
                massive_limit=args.massive_limit,
                stop_loss_frac=args.stop_loss_frac,
                trail_activate_frac=args.trail_activate_frac,
                trail_retrace_frac=args.trail_retrace_frac,
                client=client,
                serialize_ibkr=workers > 1,
                market_hours_only=args.market_hours_only,
                buffer_open_minutes=args.buffer_open_minutes,
                buffer_close_minutes=args.buffer_close_minutes,
                event_lookahead_days=args.event_lookahead_days,
                short_dte=args.short_dte,
                dte_min_override=args.dte_min,
                dte_max_override=args.dte_max,
            )
        except Exception as e:
            return {
                "symbol": sym,
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "skip_reason": str(e)[:500],
            }

    if workers == 1:
        for i, sym in enumerate(syms):
            if i:
                time.sleep(max(0.0, args.ibkr_delay))
            row = _run_sym(sym)
            results.append(row)
            st = row.get("status", "")
            sr = row.get("skip_reason", "")
            dec = row.get("decision")
            strat = dec.get("strategy_name", "") if isinstance(dec, dict) else ""
            print(f"{sym}: {st} {strat} {sr}")

            if isinstance(dec, dict) and dec.get("action") == "enter":
                time.sleep(max(0.0, args.massive_delay))
    else:
        by_sym: dict[str, dict[str, object]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_sym, sym): sym for sym in syms}
            for fut in concurrent.futures.as_completed(futs):
                sym = futs[fut]
                row = fut.result()
                by_sym[sym] = row
                st = row.get("status", "")
                sr = row.get("skip_reason", "")
                dec = row.get("decision")
                strat = dec.get("strategy_name", "") if isinstance(dec, dict) else ""
                print(f"{sym}: {st} {strat} {sr}", flush=True)
        results = [by_sym[s] for s in syms]

    actives = [r for r in results if r.get("status") == "active"]
    actives.sort(key=lambda r: float(r.get("rank_score") or 0.0), reverse=True)
    if args.top > 0 and actives:
        keep_ids = {id(x) for x in actives[: args.top]}
        for r in results:
            if r.get("status") == "active" and id(r) not in keep_ids:
                r["status"] = "trimmed"
                r["skip_reason"] = f"not_in_top_{args.top}"

    final_active = [r for r in results if r.get("status") == "active"]
    final_active.sort(key=lambda r: float(r.get("rank_score") or 0.0), reverse=True)

    # Duplicate guard: suppress new active plans for symbols that already have a CONFIRMED
    # open paper position (paper_opened=True, paper_close_sent not set) in the EXISTING
    # plans JSON file.  We check paper_opened rather than monitor state entries so that
    # hypothetical / unentered plans (where ibkr_paper_trade_from_plans failed) do not
    # trigger false duplicates.
    if not args.allow_duplicates:
        existing_plans_path = ROOT / args.out if not args.out.is_absolute() else args.out
        open_syms: set[str] = set()
        if existing_plans_path.is_file():
            try:
                prev_payload: dict = json.loads(existing_plans_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                prev_payload = {}
            for row in (prev_payload.get("active_ranked") or []) + (prev_payload.get("results") or []):
                if not isinstance(row, dict):
                    continue
                if not row.get("paper_opened"):
                    continue  # not yet confirmed open — not a real position
                if row.get("paper_close_sent"):
                    continue  # already closed
                sym_part = str(row.get("symbol", "")).upper()
                if sym_part:
                    open_syms.add(sym_part)
        for r in results:
            if r.get("status") == "active":
                sym_upper = str(r.get("symbol", "")).upper()
                if sym_upper in open_syms:
                    r["status"] = "duplicate_active"
                    r["skip_reason"] = f"symbol_already_open_in_monitor ({sym_upper})"
                    print(f"  [dup-guard] {sym_upper} already open — marking duplicate_active")
        # Rebuild final_active after dedup
        final_active = [r for r in results if r.get("status") == "active"]
        final_active.sort(key=lambda r: float(r.get("rank_score") or 0.0), reverse=True)

    out_path = ROOT / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_requested": syms,
        "results": results,
        "active_ranked": final_active,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}  (active setups: {len(actives)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
