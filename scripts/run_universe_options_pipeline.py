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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.data.liquidity_universe import LIQUID_UNIVERSE
from rlm.data.massive import MassiveClient
from rlm.data.massive_option_chain import massive_option_chain_from_client
from rlm.data.option_chain import select_nearest_expiry_slice
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.execution.risk_targets import build_spread_exit_thresholds
from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.live_model import LiveRegimeModelConfig, load_live_regime_model
from rlm.forecasting.pipeline import ForecastPipeline
from rlm.roee.chain_match import estimate_entry_cost_from_matched_legs, estimate_mark_value_from_matched_legs, match_legs_to_chain
from rlm.roee.decision import select_trade_for_row
from rlm.scoring.state_matrix import classify_state_matrix


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
    move_window: int,
    vol_window: int,
    strike_increment: float,
    attach_vix: bool,
    massive_limit: int,
    stop_loss_frac: float,
    trail_activate_frac: float,
    trail_retrace_frac: float,
    client: MassiveClient,
    live_model: LiveRegimeModelConfig | None,
) -> dict[str, object]:
    run_at = datetime.now(timezone.utc).isoformat()
    base: dict[str, object] = {
        "symbol": sym,
        "run_at_utc": run_at,
        "status": "skipped",
    }

    bars = fetch_historical_stock_bars(
        sym,
        duration=duration,
        bar_size="1 day",
        timeout_sec=120.0,
    )
    if bars.empty:
        base["skip_reason"] = "no_ibkr_bars"
        return base

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
    out["has_major_event"] = False

    last = out.iloc[-1]
    ts = last.name if isinstance(last.name, pd.Timestamp) else pd.Timestamp.utcnow().tz_localize(None)

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
        "S_D": float(last["S_D"]) if pd.notna(last["S_D"]) else None,
        "S_V": float(last["S_V"]) if pd.notna(last["S_V"]) else None,
        "S_L": float(last["S_L"]) if pd.notna(last["S_L"]) else None,
        "S_G": float(last["S_G"]) if pd.notna(last["S_G"]) else None,
    }
    base["pipeline"] = pipeline_row

    decision = select_trade_for_row(last, strike_increment=strike_increment, **decision_kwargs)
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
    expiry_slice = select_nearest_expiry_slice(chain, c.target_dte_min, c.target_dte_max)
    if expiry_slice.empty:
        base["skip_reason"] = "no_contracts_in_dte_window"
        base["dte_window"] = [c.target_dte_min, c.target_dte_max]
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
    credit_style = any(x in sn for x in ("iron_condor", "condor", "short_strangle", "strangle")) or float(entry_debit) < 0
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
    p.add_argument("--duration", default="180 D", help="IBKR history window")
    p.add_argument("--move-window", type=int, default=100)
    p.add_argument("--vol-window", type=int, default=100)
    p.add_argument("--strike-increment", type=float, default=1.0)
    p.add_argument("--ibkr-delay", type=float, default=0.35)
    p.add_argument("--massive-delay", type=float, default=0.35, help="Pause after each Massive snapshot pull")
    p.add_argument("--no-vix", action="store_true")
    p.add_argument("--massive-limit", type=int, default=250, help="Per-page limit for option snapshot pagination")
    p.add_argument("--stop-loss-frac", type=float, default=0.5, help="Hard stop = V0 - frac * entry_debit")
    p.add_argument("--trail-activate-frac", type=float, default=0.15, help="Start trailing after V0 + frac * D")
    p.add_argument("--trail-retrace-frac", type=float, default=0.25, help="Trail stop = peak * (1 - this)")
    p.add_argument("--top", type=int, default=0, help="If >0, only keep N highest rank_score among active plans")
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
    args = p.parse_args()

    try:
        client = MassiveClient()
    except ValueError as e:
        print(f"Massive client: {e}", file=sys.stderr)
        return 1

    syms = _parse_symbols(args.symbols)
    results: list[dict[str, object]] = []
    live_model: LiveRegimeModelConfig | None = None
    if not args.ignore_live_model:
        live_model_path = ROOT / args.live_model_config if not args.live_model_config.is_absolute() else args.live_model_config
        if live_model_path.is_file():
            live_model = load_live_regime_model(live_model_path)
            print(f"Using live model config: {live_model_path}")

    for i, sym in enumerate(syms):
        if i:
            time.sleep(max(0.0, args.ibkr_delay))
        try:
            row = _one_symbol(
                sym,
                duration=args.duration,
                move_window=args.move_window,
                vol_window=args.vol_window,
                strike_increment=args.strike_increment,
                attach_vix=not args.no_vix,
                massive_limit=args.massive_limit,
                stop_loss_frac=args.stop_loss_frac,
                trail_activate_frac=args.trail_activate_frac,
                trail_retrace_frac=args.trail_retrace_frac,
                client=client,
                live_model=live_model,
            )
        except Exception as e:
            row = {
                "symbol": sym,
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "skip_reason": str(e)[:500],
            }
        results.append(row)
        st = row.get("status", "")
        sr = row.get("skip_reason", "")
        dec = row.get("decision")
        strat = dec.get("strategy_name", "") if isinstance(dec, dict) else ""
        print(f"{sym}: {st} {strat} {sr}")

        if isinstance(dec, dict) and dec.get("action") == "enter":
            time.sleep(max(0.0, args.massive_delay))

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
