#!/usr/bin/env python3
"""
**Universe -> full equity pipeline -> ROEE -> Massive option chain -> match legs -> risk plan.**

For each symbol (default: ``LIQUID_UNIVERSE`` = Mag7 + SPY + QQQ):

1. IBKR bars at ``--bar-size`` / duration (overridable via ``live_regime_model.json`` → ``timeframe_hierarchy.primary_*``).
2. **Massive option snapshot** (broad chain) joined in ``prepare_bars_for_factors`` so GEX, volume/OI,
   and surface features feed factors before the regime/forecast stack (unless ``--no-chain-for-factors``).
3. Factors → state matrix → forecast/regime (``live_regime_model.model``: forecast | hmm | markov).
4. Optional **confirmation timeframes**: ``timeframe_hierarchy.confirmation_bar_sizes`` triggers separate IBKR
   fetches (short ``confirmation_duration``); downstream alignment gates ROEE when finer bars disagree with primary bias.
5. ``select_trade_for_row`` → strategy + abstract legs under regime + forecast + confirmation context.
6. If **enter**: Massive chain match + risk plan JSON.


Output: JSON file consumed by ``scripts/monitor_active_trade_plans.py``.

Examples::

    python scripts/run_universe_options_pipeline.py --out data/processed/universe_trade_plans.json
    python scripts/run_universe_options_pipeline.py --top 3 --stop-loss-frac 0.45

BLAS/OpenMP and PyTorch threads are capped automatically (see ``RLM_MAX_CPU_THREADS`` in
``.env.example``); set it before launch for a stricter ceiling on shared VPSes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import warnings
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
from rlm.data.bars_enrichment import prepare_bars_for_factors
from rlm.execution.risk_targets import build_spread_exit_thresholds
from rlm.features.factors.pipeline import FactorPipeline
from rlm.forecasting.engines import ForecastPipeline
from rlm.forecasting.live_model import (
    LiveKronosParameters,
    LiveRegimeModelConfig,
    LiveTimeframeHierarchy,
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
from rlm.roee.system_gate import SystemGate
from rlm.roee.regime_safety import attach_regime_safety_columns
from rlm.features.scoring.state_matrix import classify_state_matrix
from rlm.types.options import TradeDecision
from rlm.monitoring.structured import build_pipeline_event
from rlm.utils.market_hours import entry_window_open, session_label

# Statsmodels emits this repeatedly for our non-fixed-frequency trading calendar index.
# It is expected and buries actionable operational logs.
warnings.filterwarnings(
    "ignore",
    message="A date index has been provided, but it has no associated frequency information and so will be ignored",
)


def _env_truthy(key: str) -> bool:
    v = (os.environ.get(key) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


_IBKR_HIST_LOCK = threading.Lock()

# Keep in sync with ``scripts/monitor_active_trade_plans._TRADE_LOG_COLUMNS``.
_TRADE_LOG_COLUMNS = [
    "timestamp_utc",
    "plan_id",
    "symbol",
    "strategy",
    "entry_debit",
    "entry_mid",
    "current_mark",
    "peak_mark",
    "unrealized_pnl",
    "unrealized_pnl_pct",
    "signal",
    "closed",
    "dte",
]


def _kronos_stub_available() -> bool:
    try:
        from rlm.forecasting.models.kronos.model.kronos import KronosTokenizer

        method = getattr(KronosTokenizer, "from_pretrained", None)
        if method is None:
            return False
        consts = getattr(method, "__code__", None)
        if consts is None:
            return False
        return any("stub module" in str(c).lower() for c in consts.co_consts if isinstance(c, str))
    except Exception:
        return False


def _ensure_trade_log_with_header(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_TRADE_LOG_COLUMNS)


def _fetch_ibkr_bars(
    sym: str,
    *,
    duration: str,
    bar_size: str,
    serialize_ibkr: bool,
) -> pd.DataFrame:
    fetch_kw: dict[str, object] = {
        "duration": duration,
        "bar_size": bar_size,
        "timeout_sec": 120.0,
    }
    if serialize_ibkr:
        with _IBKR_HIST_LOCK:
            return fetch_historical_stock_bars(sym, **fetch_kw)
    return fetch_historical_stock_bars(sym, **fetch_kw)


def _factor_snapshot_params(massive_limit: int) -> dict[str, object]:
    """Wide snapshot for GEX / surface / volume–OI enrichment (no strike window)."""
    return {
        "limit": int(massive_limit),
        "sort": "expiration_date",
        "order": "asc",
    }


def _regime_key_head(regime_key: object | None) -> str:
    if regime_key is None or (isinstance(regime_key, float) and pd.isna(regime_key)):
        return ""
    return str(regime_key).split("|")[0].strip().lower()


def _finite_sd(val: object) -> float:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(x):
        return 0.0
    return float(x)


def _direction_aligned(primary_sd: float, confirm_sd: float) -> bool:
    if primary_sd == 0.0 and confirm_sd == 0.0:
        return True
    if primary_sd == 0.0 or confirm_sd == 0.0:
        return False
    return (primary_sd > 0.0 and confirm_sd > 0.0) or (primary_sd < 0.0 and confirm_sd < 0.0)


def _tf_confirmation_layers_agree(
    primary: pd.Series,
    confirm: pd.Series,
    *,
    mode: str,
) -> bool:
    direction_ok = _direction_aligned(_finite_sd(primary.get("S_D")), _finite_sd(confirm.get("S_D")))
    head_ok = _regime_key_head(primary.get("regime_key")) == _regime_key_head(confirm.get("regime_key"))
    if mode == "direction":
        return direction_ok
    if mode == "regime_head":
        return head_ok
    return direction_ok and head_ok


def _forecast_final_row(
    sym: str,
    *,
    duration: str,
    bar_size: str,
    attach_vix: bool,
    live_model: LiveRegimeModelConfig | None,
    move_window: int,
    vol_window: int,
    min_regime_train_samples: int,
    purge_bars: int,
    event_lookahead_days: int,
    serialize_ibkr: bool,
    factor_option_chain: pd.DataFrame | None = None,
) -> pd.Series | None:
    """Run bars → factors → regime/forecast stack; return latest row (for TF confirmation)."""
    bars = _fetch_ibkr_bars(sym, duration=duration, bar_size=bar_size, serialize_ibkr=serialize_ibkr)
    if bars.empty:
        return None
    df = bars.sort_values("timestamp").set_index("timestamp")
    chain = factor_option_chain if factor_option_chain is not None and not factor_option_chain.empty else None
    df = prepare_bars_for_factors(df, option_chain=chain, underlying=sym, attach_vix=attach_vix)
    feats = FactorPipeline().run(df)
    feats = classify_state_matrix(feats)
    if live_model is not None:
        forecast = live_model.build_pipeline()
    else:
        forecast = ForecastPipeline(move_window=move_window, vol_window=vol_window)
    out = forecast.run(feats).copy()
    out["has_major_event"] = bool(has_major_event_today(sym, lookahead_days=event_lookahead_days))
    out = attach_regime_safety_columns(
        out,
        min_regime_train_samples=min_regime_train_samples,
        purge_bars=purge_bars,
    )
    return out.iloc[-1]


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
    processed_dir: Path | None,
    gate: SystemGate | None = None,
    bars: pd.DataFrame | None = None,
    factor_option_chain: pd.DataFrame | None = None,
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

    raw_bars = bars if bars is not None else _fetch_ibkr_bars(sym, duration=duration, bar_size=bar_size, serialize_ibkr=serialize_ibkr)
    if raw_bars.empty:
        base["skip_reason"] = "no_ibkr_bars"
        return base, None, None

    df = raw_bars.sort_values("timestamp").set_index("timestamp")
    chain = factor_option_chain if factor_option_chain is not None and not factor_option_chain.empty else None
    df = prepare_bars_for_factors(df, option_chain=chain, underlying=sym, attach_vix=attach_vix)

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

    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        feats.to_csv(processed_dir / f"features_{sym}.csv")
        out.to_csv(processed_dir / f"forecast_features_{sym}.csv")

    last = out.iloc[-1].copy()
    ts = last.name if isinstance(last.name, pd.Timestamp) else pd.Timestamp.now(tz="UTC").tz_localize(None)

    hier = live_model.timeframe_hierarchy if live_model is not None else LiveTimeframeHierarchy()
    if hier.confirmation_bar_sizes:
        detail_map: dict[str, object] = {}
        layer_ok: list[bool] = []
        for cf_bar in hier.confirmation_bar_sizes:
            cfbs = str(cf_bar).strip()
            if not cfbs:
                continue
            cf_last = _forecast_final_row(
                sym,
                duration=str(hier.confirmation_duration).strip(),
                bar_size=cfbs,
                attach_vix=attach_vix,
                live_model=live_model,
                move_window=move_window,
                vol_window=vol_window,
                min_regime_train_samples=min_regime_train_samples,
                purge_bars=purge_bars,
                event_lookahead_days=event_lookahead_days,
                serialize_ibkr=serialize_ibkr,
                factor_option_chain=None,
            )
            if cf_last is None:
                layer_ok.append(False)
                detail_map[cfbs] = {"ok": False, "reason": "no_ibkr_bars"}
                continue
            ok = _tf_confirmation_layers_agree(last, cf_last, mode=hier.confirmation_mode)
            layer_ok.append(ok)
            detail_map[cfbs] = {
                "ok": ok,
                "regime_key": str(cf_last.get("regime_key", "")),
                "S_D": float(cf_last["S_D"]) if pd.notna(cf_last.get("S_D")) else None,
            }
        if layer_ok:
            agg = all(layer_ok) if hier.require_all_confirmations else any(layer_ok)
            last["tf_confirmation_failed"] = not agg
            last["tf_confirmation_detail"] = json.dumps(detail_map, default=str)
            last["tf_confirmation_rationale"] = (
                "Primary timeframe bias disagreed with confirmation timeframe(s) "
                f"(mode={hier.confirmation_mode})."
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
    if "tf_confirmation_failed" in last.index:
        pipeline_row["tf_confirmation_failed"] = bool(last["tf_confirmation_failed"])
        detail = last.get("tf_confirmation_detail")
        pipeline_row["tf_confirmation_detail"] = (
            str(detail) if detail is not None and pd.notna(detail) else None
        )

    base["pipeline"] = pipeline_row

    decision = select_trade_for_row(
        last,
        strike_increment=strike_increment,
        **decision_kwargs,
        regime_train_sample_count=int(last.get("regime_train_sample_count", 0) or 0),
        min_regime_train_samples=min_regime_train_samples,
        regime_purge_bars=purge_bars,
        short_dte=short_dte,
        gate=gate,
    )
    base["decision"] = {
        "action": decision.action,
        "strategy_name": decision.strategy_name,
        "rationale": decision.rationale,
        "size_fraction": decision.size_fraction,
        "regime_key": decision.regime_key,
        "metadata": {k: v for k, v in (decision.metadata or {}).items() if k != "matched_legs"},
    }
    event = build_pipeline_event(
        symbol=sym,
        bar_id=str(ts),
        factor_values={
            "S_D": pipeline_row["S_D"],
            "S_V": pipeline_row["S_V"],
            "S_L": pipeline_row["S_L"],
            "S_G": pipeline_row["S_G"],
        },
        regime_state=str(pipeline_row.get("regime_key") or ""),
        kronos_confidence=(
            float(last["kronos_confidence"]) if "kronos_confidence" in last and pd.notna(last["kronos_confidence"]) else None
        ),
        action=decision.action,
        extra={"strategy_name": decision.strategy_name},
    )
    print(json.dumps({"event": "pipeline_bar", **event}, default=str), flush=True)

    if decision.action != "enter" or not decision.candidate or not decision.legs:
        if decision.strategy_name == "regime_safety_check":
            base["skip_reason"] = "regime_safety_check"
        elif decision.strategy_name == "timeframe_confirmation_block":
            base["skip_reason"] = "timeframe_confirmation_block"
        else:
            base["skip_reason"] = "roee_skip_or_no_legs"
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
        params["expiration_date.gte"] = str((anchor + pd.Timedelta(days=int(candidate.target_dte_min))).date())
        params["expiration_date.lte"] = str((anchor + pd.Timedelta(days=int(candidate.target_dte_max))).date())

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

    dte_min = int(dte_min_override) if dte_min_override is not None else int(candidate.target_dte_min)
    dte_max = int(dte_max_override) if dte_max_override is not None else int(candidate.target_dte_max)
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


def _strategy_name_for_row(row: dict[str, object]) -> str:
    strategy_name = str(row.get("strategy_name") or row.get("strategy") or "").strip()
    if strategy_name:
        return strategy_name
    dec = row.get("decision")
    if isinstance(dec, dict):
        return str(dec.get("strategy_name") or dec.get("strategy") or "").strip()
    return ""


def _load_open_symbols_from_trade_log(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    latest_by_plan: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("plan_id") or "").strip()
                if pid:
                    latest_by_plan[pid] = {k: str(v) for k, v in row.items()}
    except OSError:
        return set()
    open_symbols: set[str] = set()
    for row in latest_by_plan.values():
        if str(row.get("closed") or "0").strip() == "1":
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if sym:
            open_symbols.add(sym)
    return open_symbols


def _apply_active_plan_guards(
    results: list[dict[str, object]],
    *,
    max_active_per_symbol: int,
    open_symbols: set[str],
) -> None:
    active_rows = [r for r in results if r.get("status") == "active"]
    active_rows.sort(key=lambda r: float(r.get("rank_score") or 0.0), reverse=True)
    seen_symbol_strategy: set[tuple[str, str]] = set()
    active_by_symbol: dict[str, int] = {}

    for row in active_rows:
        sym = str(row.get("symbol") or "").strip().upper()
        strategy = _strategy_name_for_row(row)

        if sym and sym in open_symbols:
            row["status"] = "trimmed"
            row["skip_reason"] = "symbol_already_open_in_trade_log"
            continue

        key = (sym, strategy)
        if key in seen_symbol_strategy:
            row["status"] = "trimmed"
            row["skip_reason"] = "duplicate_symbol_strategy_or_max_active_per_symbol"
            continue

        n_active = active_by_symbol.get(sym, 0)
        if sym and n_active >= int(max_active_per_symbol):
            row["status"] = "trimmed"
            row["skip_reason"] = "duplicate_symbol_strategy_or_max_active_per_symbol"
            continue

        seen_symbol_strategy.add(key)
        if sym:
            active_by_symbol[sym] = n_active + 1


def _write_universe_latest_views(final_results: list[dict[str, object]], out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for row in final_results:
        pipeline = row.get("pipeline")
        if not isinstance(pipeline, dict):
            continue
        decision = row.get("decision")
        rows.append(
            {
                "run_at_utc": row.get("run_at_utc"),
                "symbol": row.get("symbol"),
                "status": row.get("status"),
                "skip_reason": row.get("skip_reason"),
                "action": decision.get("action") if isinstance(decision, dict) else None,
                "strategy_name": decision.get("strategy_name") if isinstance(decision, dict) else None,
                "regime_key": pipeline.get("regime_key"),
                "close": pipeline.get("close"),
                "sigma": pipeline.get("sigma"),
                "S_D": pipeline.get("S_D"),
                "S_V": pipeline.get("S_V"),
                "S_L": pipeline.get("S_L"),
                "S_G": pipeline.get("S_G"),
                "regime_safety_ok": pipeline.get("regime_safety_ok"),
                "regime_train_sample_count": pipeline.get("regime_train_sample_count"),
            }
        )

    if rows:
        df = pd.DataFrame(rows).sort_values(["symbol", "run_at_utc"]).reset_index(drop=True)
    else:
        fallback_rows = [
            {
                "run_at_utc": row.get("run_at_utc"),
                "symbol": row.get("symbol"),
                "status": row.get("status"),
                "skip_reason": row.get("skip_reason"),
                "action": None,
                "strategy_name": None,
                "regime_key": None,
                "close": None,
                "sigma": None,
                "S_D": None,
                "S_V": None,
                "S_L": None,
                "S_G": None,
                "regime_safety_ok": None,
                "regime_train_sample_count": None,
            }
            for row in final_results
        ]
        df = pd.DataFrame(fallback_rows).sort_values(["symbol", "run_at_utc"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "universe_forecast_latest.csv", index=False)
    try:
        df.to_parquet(out_dir / "universe_forecast_latest.parquet", index=False)
    except Exception:
        pass

    symbol_index = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": sorted(df["symbol"].dropna().astype(str).str.upper().unique().tolist()),
        "rows": int(len(df)),
    }
    (out_dir / "universe_symbol_index.json").write_text(
        json.dumps(symbol_index, indent=2, default=str),
        encoding="utf-8",
    )


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
        "--max-active-per-symbol",
        type=int,
        default=1,
        help="Maximum active plans kept per symbol after rank sorting.",
    )
    p.add_argument(
        "--respect-open-trade-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim new plans for symbols that already have open rows in --trade-log.",
    )
    p.add_argument(
        "--trade-log",
        type=Path,
        default=Path("data/processed/trade_log.csv"),
        help="Trade log CSV used for open-position gating when --respect-open-trade-log is enabled.",
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
        default=0,
        help="Pause new trades when the current regime has fewer prior training samples than this threshold.",
    )
    p.add_argument(
        "--chain-for-factors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch Massive option snapshots before factor prep (GEX, volume/OI, surface features).",
    )
    p.add_argument(
        "--live-model-config",
        type=Path,
        default=Path("data/processed/live_regime_model.json"),
        help="Optional promoted live-model JSON. Falls back to ForecastPipeline if missing.",
    )
    p.add_argument("--ignore-live-model", action="store_true", help="Ignore any promoted live model config.")
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

    try:
        client = MassiveClient()
    except ValueError as e:
        print(f"Massive client: {e}", file=sys.stderr)
        return 1

    syms = _parse_symbols(args.symbols)
    processed_dir = (ROOT / "data" / "processed").resolve()
    live_model: LiveRegimeModelConfig | None = None
    live_model_bootstrapped = False
    live_model_path: Path | None = None
    if not args.ignore_live_model:
        live_model_path = (
            ROOT / args.live_model_config if not args.live_model_config.is_absolute() else args.live_model_config
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
                f"[live_model] {live_model_path} missing - using defaults; "
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
        print(f"[kronos] Blend enabled - weight={args.kronos_weight}, stride={args.kronos_stride}")
    if live_model is not None and bool(live_model.use_kronos) and _kronos_stub_available():
        live_model = live_model.model_copy(update={"use_kronos": False})
        print("[kronos] Disabled for this run: vendored Kronos runtime is stub-only on this host.", flush=True)
    if live_model is not None:
        live_model = apply_nightly_hyperparam_overlay(live_model, ROOT)
    min_regime_train_samples = int(args.min_regime_train_samples)
    if live_model is not None and live_model.min_regime_train_samples is not None:
        min_regime_train_samples = int(live_model.min_regime_train_samples)

    if (
        live_model is not None
        and live_model_bootstrapped
        and live_model_path is not None
        and not args.ignore_live_model
    ):
        save_live_regime_model(live_model, live_model_path)
        print(f"[live_model] saved bootstrap config to {live_model_path}")

    duration = str(args.duration)
    bar_size = str(args.bar_size)
    if live_model is not None:
        th = live_model.timeframe_hierarchy
        if th.primary_duration:
            pdur = str(th.primary_duration).strip()
            if pdur:
                duration = pdur
        if th.primary_bar_size:
            pbs = str(th.primary_bar_size).strip()
            if pbs:
                bar_size = pbs
    if bar_size != "1 day" and duration.endswith(" D"):
        try:
            days = int(duration.split()[0])
            if days < 30:
                duration = "30 D"
        except (ValueError, IndexError):
            pass

    hot_cache_symbols = _parse_symbols(args.massive_hot_cache_symbols)
    # When Hermes (or manual edits) sets STAND-DOWN, ROEE returns system_gate_block for every symbol.
    # Paper hosts often want the quant pipeline independent of LLM posture; set RLM_SKIP_SYSTEM_GATE=1.
    gate: SystemGate | None = None if _env_truthy("RLM_SKIP_SYSTEM_GATE") else SystemGate(ROOT)
    if gate is None:
        print("[gate] RLM_SKIP_SYSTEM_GATE=1 — ROEE ignores data/processed/gate_state.json", flush=True)
    trade_log_path = ROOT / args.trade_log if not args.trade_log.is_absolute() else args.trade_log
    _ensure_trade_log_with_header(trade_log_path)

    chain_for_factors = bool(args.chain_for_factors) and not _env_truthy("RLM_NO_CHAIN_FOR_FACTORS")
    if not chain_for_factors:
        print(
            "[chain] factor-stage Massive snapshots disabled (--no-chain-for-factors or RLM_NO_CHAIN_FOR_FACTORS)",
            flush=True,
        )

    results: list[dict[str, object] | None] = [None] * len(syms)
    pending: list[_PendingUniverseSymbol] = []

    def _outside_reason() -> str | None:
        if not bool(args.market_hours_only):
            return None
        if entry_window_open(
            buffer_open_minutes=int(args.buffer_open_minutes),
            buffer_close_minutes=int(args.buffer_close_minutes),
        ):
            return None
        return f"outside_entry_window ({session_label()})"

    bars_by_index: list[pd.DataFrame | None] = [None] * len(syms)
    for i, sym in enumerate(syms):
        orsn = _outside_reason()
        if orsn is not None:
            results[i] = {
                "symbol": sym,
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "skipped",
                "skip_reason": orsn,
            }
            continue
        if i:
            time.sleep(max(0.0, args.ibkr_delay))
        try:
            bdf = _fetch_ibkr_bars(sym, duration=duration, bar_size=bar_size, serialize_ibkr=bool(args.serialize_ibkr))
            if bdf.empty:
                results[i] = {
                    "symbol": sym,
                    "run_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "skipped",
                    "skip_reason": "no_ibkr_bars",
                }
                continue
            bars_by_index[i] = bdf
        except Exception as e:
            results[i] = {
                "symbol": sym,
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "skip_reason": str(e)[:500],
            }

    factor_chains: dict[str, pd.DataFrame] = {}
    if chain_for_factors:
        prefetch_syms: list[str] = []
        ts_map: dict[str, pd.Timestamp] = {}
        for i, sym in enumerate(syms):
            bdf = bars_by_index[i]
            if bdf is None:
                continue
            df_ix = bdf.sort_values("timestamp").set_index("timestamp")
            prefetch_syms.append(sym)
            ts_map[sym] = df_ix.index[-1]
        if prefetch_syms:
            fp = _factor_snapshot_params(args.massive_limit)
            batch_fc = massive_option_chains_from_client(
                client,
                prefetch_syms,
                timestamps=ts_map,
                max_workers=max(1, int(args.massive_workers)),
                cache_ttl_s=0.0,
                **fp,
            )
            factor_chains = dict(batch_fc.chains)
            for esym, err in batch_fc.errors.items():
                print(f"[chain] factor-stage Massive error {esym}: {err}", flush=True)

    for i, sym in enumerate(syms):
        if results[i] is not None:
            continue
        bdf = bars_by_index[i]
        if bdf is None:
            continue
        try:
            base, decision, ts = _prepare_symbol(
                sym,
                duration=duration,
                bar_size=bar_size,
                move_window=args.move_window,
                vol_window=args.vol_window,
                strike_increment=args.strike_increment,
                attach_vix=not args.no_vix,
                min_regime_train_samples=min_regime_train_samples,
                purge_bars=args.purge_bars,
                live_model=live_model,
                market_hours_only=bool(args.market_hours_only),
                buffer_open_minutes=int(args.buffer_open_minutes),
                buffer_close_minutes=int(args.buffer_close_minutes),
                event_lookahead_days=int(args.event_lookahead_days),
                serialize_ibkr=bool(args.serialize_ibkr),
                short_dte=bool(args.short_dte),
                processed_dir=processed_dir,
                gate=gate,
                bars=bdf,
                factor_option_chain=factor_chains.get(sym),
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
    open_symbols = _load_open_symbols_from_trade_log(trade_log_path) if args.respect_open_trade_log else set()
    _apply_active_plan_guards(
        final_results,
        max_active_per_symbol=max(1, int(args.max_active_per_symbol)),
        open_symbols=open_symbols,
    )
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
    _write_universe_latest_views(final_results, processed_dir)
    print(f"\nWrote {out_path}  (active setups: {len(final_active)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
