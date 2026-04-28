#!/usr/bin/env python3
"""
**Universe → full equity pipeline → ROEE → Massive option chain → match legs → risk plan.**

For each symbol (default: ``LIQUID_UNIVERSE`` = Mag7 + SPY + QQQ):

1. IBKR daily bars → ``prepare_bars_for_factors`` → factors → state matrix → forecast (latest bar).
2. ``select_trade_for_row`` → strategy + abstract legs.
3. If **enter**: fetch a filtered Massive option snapshot (paginated), normalize,
   DTE slice, ``match_legs_to_chain``.
4. Build **entry debit** (bid/ask), **mid mark** ``V0``, take-profit /
   hard-stop / trailing activation levels
   (:mod:`rlm.execution.risk_targets`), plus JSON suitable for ``ibkr_place_roee_combo.py``.

Output: JSON file consumed by ``scripts/monitor_active_trade_plans.py``.

Examples::

    python scripts/run_universe_options_pipeline.py --out data/processed/universe_trade_plans.json
    python scripts/run_universe_options_pipeline.py --top 3 --stop-loss-frac 0.45

BLAS/OpenMP and PyTorch threads are capped automatically (see ``RLM_MAX_CPU_THREADS`` in
``.env.example``); set it before launch for a stricter ceiling on shared VPSes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.utils.compute_threads import apply_compute_thread_env  # noqa: E402

apply_compute_thread_env()

import numpy as np
import pandas as pd

# ruff: noqa: E402
from rlm.data.event_calendar import has_major_event_today
from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.data.liquidity_universe import LIQUID_UNIVERSE
from rlm.data.massive import MassiveClient
from rlm.data.massive_option_chain import massive_option_chains_from_client
from rlm.data.option_chain import select_nearest_expiry_slice
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.execution.risk_targets import build_spread_exit_thresholds
from rlm.factors import FactorPipeline
from rlm.forecasting.engines import ForecastPipeline
from rlm.forecasting.live_model import (
    LiveKronosParameters,
    LiveRegimeModelConfig,
    apply_nightly_hyperparam_overlay,
    load_live_regime_model,
    save_live_regime_model,
)
from rlm.roee.chain_match import (
    estimate_entry_cost_from_matched_legs,
    estimate_mark_value_from_matched_legs,
    match_legs_to_chain,
)
from rlm.roee.decision import select_trade_for_row
from rlm.roee.regime_safety import attach_regime_safety_columns
from rlm.scoring.state_matrix import classify_state_matrix
from rlm.types.options import TradeDecision
from rlm.utils.market_hours import entry_window_open, session_label

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


@dataclass
class _PendingUniverseSymbol:
    index: int
    symbol: str
    base: dict[str, object]
    decision: TradeDecision
    timestamp: pd.Timestamp


def _prepare_symbol(
    sym: str,
    *,
    duration: str,
    bar_size: str,
    move_window: int,
    vol_window: int,
    strike_increment: float,
    attach_vix: bool,
    min_regime_train_samples: int,
    purge_bars: int,
    live_model: LiveRegimeModelConfig | None,
    market_hours_only: bool,
    buffer_open_minutes: int,
    buffer_close_minutes: int,
    event_lookahead_days: int,
    serialize_ibkr: bool,
    short_dte: bool,
) -> tuple[dict[str, object], TradeDecision | None, pd.Timestamp | None]:
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
        return base, None, None

    fetch_kw: dict[str, object] = {
        "duration": duration,
        "bar_size": bar_size,
        "timeout_sec": 120.0,
    }
    if serialize_ibkr:
        with _IBKR_HIST_LOCK:
            bars = fetch_historical_stock_bars(sym, **fetch_kw)
    else:
        bars = fetch_historical_stock_bars(sym, **fetch_kw)
    if bars.empty:
        base["skip_reason"] = "no_ibkr_bars"
        return base, None, None

    df = bars.sort_values("timestamp").set_index("timestamp")
    df = prepare_bars_for_factors(df, option_chain=None, underlying=sym, attach_vix=attach_vix)

    feats = FactorPipeline().run(df)
    feats = classify_state_matrix(feats)
    if live_model is not None:
        forecast = live_model.build_pipeline()
        decision_kwargs = live_model.decision_kwargs()
        active_model = live_model.model
    else:
        forecast = ForecastPipeline(move_window=move_window, vol_window=vol_window)
        decision_kwargs = {}
        active_model = "forecast"
    out = forecast.run(feats)
    out = out.copy()
    out["has_major_event"] = bool(has_major_event_today(sym, lookahead_days=event_lookahead_days))
    out = attach_regime_safety_columns(
        out,
        min_regime_train_samples=min_regime_train_samples,
        purge_bars=purge_bars,
    )

    last = out.iloc[-1]
    ts = (
        last.name
        if isinstance(last.name, pd.Timestamp)
        else pd.Timestamp.now(tz="UTC").tz_localize(None)
    )

    pipeline_row = {
        "live_model": active_model,
        "close": float(last["close"]),
        "sigma": float(last["sigma"]) if pd.notna(last["sigma"]) else None,
        "mean_price": float(last["mean_price"]) if pd.notna(last.get("mean_price")) else None,
        "lower_1s": float(last["lower_1s"]) if pd.notna(last.get("lower_1s")) else None,
        "upper_1s": float(last["upper_1s"]) if pd.notna(last.get("upper_1s")) else None,
        "lower_2s": float(last["lower_2s"]) if pd.notna(last.get("lower_2s")) else None,
        "upper_2s": float(last["upper_2s"]) if pd.notna(last.get("upper_2s")) else None,
        "regime_key": str(last.get("regime_key", "")),
        "regime_train_sample_count": int(last.get("regime_train_sample_count", 0) or 0),
        "regime_safety_ok": bool(last.get("regime_safety_ok", True)),
        "S_D": float(last["S_D"]) if pd.notna(last["S_D"]) else None,
        "S_V": float(last["S_V"]) if pd.notna(last["S_V"]) else None,
        "S_L": float(last["S_L"]) if pd.notna(last["S_L"]) else None,
        "S_G": float(last["S_G"]) if pd.notna(last["S_G"]) else None,
    }
    base["pipeline"] = pipeline_row

    decision = select_trade_for_row(
        last,
        strike_increment=strike_increment,
        **decision_kwargs,
        regime_train_sample_count=int(last.get("regime_train_sample_count", 0) or 0),
        min_regime_train_samples=min_regime_train_samples,
        regime_purge_bars=purge_bars,
        short_dte=short_dte,
    )
    base["decision"] = {
        "action": decision.action,
        "strategy_name": decision.strategy_name,
        "rationale": decision.rationale,
        "size_fraction": decision.size_fraction,
        "regime_key": decision.regime_key,
        "metadata": {k: v for k, v in (decision.metadata or {}).items() if k != "matched_legs"},
    }

    if decision.action != "enter" or not decision.candidate or not decision.legs:
        base["skip_reason"] = (
            "regime_safety_check"
            if decision.strategy_name == "regime_safety_check"
            else "roee_skip_or_no_legs"
        )
        return base, None, None
    return base, decision, ts


def _build_incremental_snapshot_params(
    ts: pd.Timestamp,
    decision: TradeDecision,
    *,
    massive_limit: int,
    strike_increment: float,
) -> dict[str, object]:
    params: dict[str, object] = {
        "limit": int(massive_limit),
        "sort": "expiration_date",
        "order": "asc",
    }
    candidate = decision.candidate
    if candidate is not None:
        anchor = pd.Timestamp(ts)
        if anchor.tzinfo is not None:
            anchor = anchor.tz_localize(None)
        anchor = anchor.normalize()
        params["expiration_date.gte"] = str(
            (anchor + pd.Timedelta(days=int(candidate.target_dte_min))).date()
        )
        params["expiration_date.lte"] = str(
            (anchor + pd.Timedelta(days=int(candidate.target_dte_max))).date()
        )

    strikes = [float(leg.strike) for leg in decision.legs]
    if strikes:
        pad = max(float(strike_increment), 1.0)
        params["strike_price.gte"] = max(0.0, min(strikes) - pad)
        params["strike_price.lte"] = max(strikes) + pad

    option_types = {str(leg.option_type).lower().strip() for leg in decision.legs}
    option_types.discard("")
    if len(option_types) == 1:
        params["contract_type"] = next(iter(option_types))
    return params


def _finalize_symbol(
    base: dict[str, object],
    *,
    decision: TradeDecision,
    chain: pd.DataFrame,
    stop_loss_frac: float,
    trail_activate_frac: float,
    trail_retrace_frac: float,
    dte_min_override: int | None = None,
    dte_max_override: int | None = None,
) -> dict[str, object]:
    sym = str(base.get("symbol", ""))
    candidate = decision.candidate
    if candidate is None:
        base["skip_reason"] = "roee_skip_or_no_legs"
        return base

    chain = _filter_tradable_chain(chain)
    if chain.empty:
        base["skip_reason"] = "no_tradable_option_quotes"
        return base

    dte_min = (
        int(dte_min_override) if dte_min_override is not None else int(candidate.target_dte_min)
    )
    dte_max = (
        int(dte_max_override) if dte_max_override is not None else int(candidate.target_dte_max)
    )
    expiry_slice = select_nearest_expiry_slice(chain, dte_min, dte_max)
    if expiry_slice.empty:
        base["skip_reason"] = "no_contracts_in_dte_window"
        base["dte_window"] = [dte_min, dte_max]
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

    tp_pct = float(candidate.target_profit_pct)
    debit_mag = abs(float(entry_debit))
    thresholds = build_spread_exit_thresholds(
        v0=float(v0),
        entry_debit=debit_mag,
        target_profit_pct=tp_pct,
        stop_loss_frac_of_debit=stop_loss_frac,
        trail_activate_frac_of_debit=trail_activate_frac,
        trail_retrace_frac_from_peak=trail_retrace_frac,
    )

    credit_style = float(entry_debit) < 0
    combo_order_action = "SELL" if credit_style else "BUY"
    lim = float(round(debit_mag, 4))

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

    meta = decision.metadata or {}
    conf = float(meta.get("regime_confidence") or meta.get("confidence") or 0.0)
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
            "candidate": {
                "target_dte_min": candidate.target_dte_min,
                "target_dte_max": candidate.target_dte_max,
                "target_profit_pct": candidate.target_profit_pct,
                "max_risk_pct": candidate.max_risk_pct,
            },
            "regime_key": str(decision.regime_key or ""),
            "ibkr_combo_spec": ibkr_spec,
            "rank_score": score,
        }
    )
    return base


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols", default=",".join(LIQUID_UNIVERSE), help="Comma-separated tickers")
    p.add_argument("--duration", default="180 D", help="IBKR history window")
    p.add_argument(
        "--bar-size",
        default="1 day",
        help='IBKR bar size (e.g. "1 day", "1 hour")',
    )
    p.add_argument(
        "--market-hours-only",
        action="store_true",
        help="Skip symbols outside the RTH entry window (see buffer flags)",
    )
    p.add_argument(
        "--buffer-open-minutes",
        type=int,
        default=15,
        help="Minutes after 09:30 ET before allowing entries",
    )
    p.add_argument(
        "--buffer-close-minutes",
        type=int,
        default=30,
        help="Minutes before 16:00 ET after which entries are blocked",
    )
    p.add_argument(
        "--event-lookahead-days",
        type=int,
        default=1,
        help="Forward days for earnings/macro event check (has_major_event)",
    )
    p.add_argument(
        "--serialize-ibkr",
        action="store_true",
        help="Serialize IBKR historical requests (use with parallel workers)",
    )
    p.add_argument(
        "--short-dte",
        action="store_true",
        help="Use 0DTE/1DTE strategy map via select_trade(short_dte=True)",
    )
    p.add_argument(
        "--dte-min",
        type=int,
        default=None,
        help="Override min DTE when slicing Massive chain (default: candidate window)",
    )
    p.add_argument(
        "--dte-max",
        type=int,
        default=None,
        help="Override max DTE when slicing Massive chain (default: candidate window)",
    )
    p.add_argument("--move-window", type=int, default=100)
    p.add_argument("--vol-window", type=int, default=100)
    p.add_argument("--strike-increment", type=float, default=1.0)
    p.add_argument("--ibkr-delay", type=float, default=0.35)
    p.add_argument(
        "--massive-delay",
        type=float,
        default=0.35,
        help="Pause between serial Massive pulls; ignored when --massive-workers > 1",
    )
    p.add_argument("--no-vix", action="store_true")
    p.add_argument(
        "--massive-limit",
        type=int,
        default=250,
        help="Per-page limit for option snapshot pagination",
    )
    p.add_argument(
        "--massive-workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
        help="Concurrent Massive chain fetch workers (default: min(4, max(1, cpu_count//2)))",
    )
    p.add_argument(
        "--massive-cache-ttl",
        type=float,
        default=15.0,
        help="In-memory TTL in seconds for hot Massive chains",
    )
    p.add_argument(
        "--massive-hot-cache-symbols",
        default="SPY,QQQ",
        help="Comma-separated symbols eligible for RAM chain caching",
    )
    p.add_argument(
        "--stop-loss-frac",
        type=float,
        default=0.5,
        help="Hard stop = V0 - frac * entry_debit",
    )
    p.add_argument(
        "--trail-activate-frac",
        type=float,
        default=0.15,
        help="Start trailing after V0 + frac * D",
    )
    p.add_argument(
        "--trail-retrace-frac",
        type=float,
        default=0.25,
        help="Trail stop = peak * (1 - this)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        help="If >0, only keep N highest rank_score among active plans",
    )
    p.add_argument(
        "--purge-bars",
        type=int,
        default=0,
        help="Exclude the most recent bars from regime training counts.",
    )
    p.add_argument(
        "--min-regime-train-samples",
        type=int,
        default=5,
        help="Pause new trades when the current regime has fewer prior training samples than this threshold.",
    )
    p.add_argument(
        "--live-model-config",
        type=Path,
        default=Path("data/processed/live_regime_model.json"),
        help="Optional promoted live-model JSON. Falls back to ForecastPipeline if missing.",
    )
    p.add_argument(
        "--ignore-live-model", action="store_true", help="Ignore any promoted live model config."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/universe_trade_plans.json"),
        help="Output JSON (under repo root if relative)",
    )
    # Kronos foundation-model blend
    p.add_argument(
        "--use-kronos",
        action="store_true",
        help="Blend Kronos foundation-model return forecasts into every symbol's pipeline output.",
    )
    p.add_argument(
        "--kronos-weight",
        type=float,
        default=0.35,
        help="Blend weight for Kronos (0=base only, 1=Kronos only, default 0.35).",
    )
    p.add_argument("--kronos-model", default="NeoQuasar/Kronos-small")
    p.add_argument("--kronos-stride", type=int, default=1)
    p.add_argument("--kronos-samples", type=int, default=5)
    args = p.parse_args()

    duration = str(args.duration)
    bar_size = str(args.bar_size)
    if bar_size != "1 day" and duration.endswith(" D"):
        try:
            days = int(duration.split()[0])
            if days < 30:
                duration = "30 D"
        except (ValueError, IndexError):
            pass

    try:
        client = MassiveClient()
    except ValueError as e:
        print(f"Massive client: {e}", file=sys.stderr)
        return 1

    syms = _parse_symbols(args.symbols)
    live_model: LiveRegimeModelConfig | None = None
    live_model_bootstrapped = False
    live_model_path: Path | None = None
    if not args.ignore_live_model:
        live_model_path = (
            ROOT / args.live_model_config
            if not args.live_model_config.is_absolute()
            else args.live_model_config
        )
        if live_model_path.is_file():
            live_model = load_live_regime_model(live_model_path)
            print(f"Using live model config: {live_model_path}")
        else:
            # Same defaults as versioned data/processed/live_regime_model.json; persist after overlay
            # so hosts without that file match fresh-clone behavior. Weekly calibrate overwrites when run.
            live_model = LiveRegimeModelConfig(model="hmm")
            live_model_bootstrapped = True
            print(
                f"[live_model] {live_model_path} missing — using defaults; "
                "saving after nightly overlay (run calibrate_regime_models.py to tune)"
            )
    if args.use_kronos:
        kronos_params = LiveKronosParameters(
            model_name=args.kronos_model,
            stride=args.kronos_stride,
            sample_count=args.kronos_samples,
            weight=args.kronos_weight,
        )
        if live_model is not None:
            live_model = live_model.model_copy(update={"use_kronos": True, "kronos": kronos_params})
        else:
            live_model = LiveRegimeModelConfig(use_kronos=True, kronos=kronos_params)
        print(f"[kronos] Blend enabled — weight={args.kronos_weight}, stride={args.kronos_stride}")
    if live_model is not None:
        live_model = apply_nightly_hyperparam_overlay(live_model, ROOT)
    if (
        live_model is not None
        and live_model_bootstrapped
        and live_model_path is not None
        and not args.ignore_live_model
    ):
        save_live_regime_model(live_model, live_model_path)
        print(f"[live_model] saved bootstrap config to {live_model_path}")
    hot_cache_symbols = _parse_symbols(args.massive_hot_cache_symbols)
    results: list[dict[str, object] | None] = [None] * len(syms)
    pending: list[_PendingUniverseSymbol] = []

    for i, sym in enumerate(syms):
        if i:
            time.sleep(max(0.0, args.ibkr_delay))
        try:
            base, decision, ts = _prepare_symbol(
                sym,
                duration=duration,
                bar_size=bar_size,
                move_window=args.move_window,
                vol_window=args.vol_window,
                strike_increment=args.strike_increment,
                attach_vix=not args.no_vix,
                min_regime_train_samples=args.min_regime_train_samples,
                purge_bars=args.purge_bars,
                live_model=live_model,
                market_hours_only=bool(args.market_hours_only),
                buffer_open_minutes=int(args.buffer_open_minutes),
                buffer_close_minutes=int(args.buffer_close_minutes),
                event_lookahead_days=int(args.event_lookahead_days),
                serialize_ibkr=bool(args.serialize_ibkr),
                short_dte=bool(args.short_dte),
            )
        except Exception as e:
            results[i] = {
                "symbol": sym,
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "skip_reason": str(e)[:500],
            }
            continue

        if decision is None or ts is None:
            results[i] = base
            continue

        pending.append(
            _PendingUniverseSymbol(
                index=i,
                symbol=sym,
                base=base,
                decision=decision,
                timestamp=ts,
            )
        )

    if pending:
        if int(args.massive_workers) <= 1 and len(pending) > 1 and float(args.massive_delay) > 0:
            for j, item in enumerate(pending):
                if j:
                    time.sleep(max(0.0, float(args.massive_delay)))
                batch = massive_option_chains_from_client(
                    client,
                    [item.symbol],
                    timestamps={item.symbol: item.timestamp},
                    per_symbol_params={
                        item.symbol: _build_incremental_snapshot_params(
                            item.timestamp,
                            item.decision,
                            massive_limit=args.massive_limit,
                            strike_increment=args.strike_increment,
                        )
                    },
                    max_workers=1,
                    cache_ttl_s=max(0.0, float(args.massive_cache_ttl)),
                    hot_cache_symbols=hot_cache_symbols,
                )
                if item.symbol in batch.errors:
                    item.base["skip_reason"] = f"massive_chain_error:{batch.errors[item.symbol]}"
                    results[item.index] = item.base
                    continue
                chain = batch.chains.get(item.symbol)
                if chain is None:
                    item.base["skip_reason"] = "massive_chain_error:missing_chain"
                    results[item.index] = item.base
                    continue
                results[item.index] = _finalize_symbol(
                    item.base,
                    decision=item.decision,
                    chain=chain,
                    stop_loss_frac=args.stop_loss_frac,
                    trail_activate_frac=args.trail_activate_frac,
                    trail_retrace_frac=args.trail_retrace_frac,
                    dte_min_override=args.dte_min,
                    dte_max_override=args.dte_max,
                )
        else:
            batch = massive_option_chains_from_client(
                client,
                [item.symbol for item in pending],
                timestamps={item.symbol: item.timestamp for item in pending},
                per_symbol_params={
                    item.symbol: _build_incremental_snapshot_params(
                        item.timestamp,
                        item.decision,
                        massive_limit=args.massive_limit,
                        strike_increment=args.strike_increment,
                    )
                    for item in pending
                },
                max_workers=max(1, int(args.massive_workers)),
                cache_ttl_s=max(0.0, float(args.massive_cache_ttl)),
                hot_cache_symbols=hot_cache_symbols,
            )
            for item in pending:
                if item.symbol in batch.errors:
                    item.base["skip_reason"] = f"massive_chain_error:{batch.errors[item.symbol]}"
                    results[item.index] = item.base
                    continue
                chain = batch.chains.get(item.symbol)
                if chain is None:
                    item.base["skip_reason"] = "massive_chain_error:missing_chain"
                    results[item.index] = item.base
                    continue
                results[item.index] = _finalize_symbol(
                    item.base,
                    decision=item.decision,
                    chain=chain,
                    stop_loss_frac=args.stop_loss_frac,
                    trail_activate_frac=args.trail_activate_frac,
                    trail_retrace_frac=args.trail_retrace_frac,
                    dte_min_override=args.dte_min,
                    dte_max_override=args.dte_max,
                )

    final_results: list[dict[str, object]] = []
    for sym, row in zip(syms, results):
        if row is None:
            row = {
                "symbol": sym,
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "skip_reason": "missing_pipeline_result",
            }
        final_results.append(row)
        st = row.get("status", "")
        sr = row.get("skip_reason", "")
        dec = row.get("decision")
        strat = dec.get("strategy_name", "") if isinstance(dec, dict) else ""
        print(f"{sym}: {st} {strat} {sr}")

    actives = [r for r in final_results if r.get("status") == "active"]
    actives.sort(key=lambda r: float(r.get("rank_score") or 0.0), reverse=True)
    if args.top > 0 and actives:
        keep_ids = {id(x) for x in actives[: args.top]}
        for r in final_results:
            if r.get("status") == "active" and id(r) not in keep_ids:
                r["status"] = "trimmed"
                r["skip_reason"] = f"not_in_top_{args.top}"

    final_active = [r for r in final_results if r.get("status") == "active"]
    final_active.sort(key=lambda r: float(r.get("rank_score") or 0.0), reverse=True)

    out_path = ROOT / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_requested": syms,
        "results": final_results,
        "active_ranked": final_active,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}  (active setups: {len(actives)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
