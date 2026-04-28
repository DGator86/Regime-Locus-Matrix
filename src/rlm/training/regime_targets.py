from __future__ import annotations

from typing import Mapping


def derive_outcome_regime_label(strategy_target_row: Mapping[str, float]) -> str:
    scores = {k: float(v) for k, v in strategy_target_row.items()}
    defaults = [
        "bull_call_spread",
        "bear_put_spread",
        "iron_condor",
        "calendar_spread",
        "debit_spread",
        "no_trade",
    ]
    for key in defaults:
        scores.setdefault(key, 0.0)

    non_no_trade = [scores[k] for k in defaults if k != "no_trade"]
    if all(v < -0.25 for v in non_no_trade):
        return "no_trade"

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else -1e9
    margin = top_score - second_score

    if margin < 0.08:
        if scores["no_trade"] >= max(non_no_trade):
            return "no_trade"
        if scores["iron_condor"] >= 0 and scores["bull_call_spread"] * scores["bear_put_spread"] < 0:
            return "mean_reversion"

    mapping = {
        "bull_call_spread": "trend_up_stable",
        "bear_put_spread": "trend_down_stable",
        "iron_condor": "range_compression",
        "calendar_spread": "transition",
        "debit_spread": "breakout_expansion",
        "no_trade": "no_trade",
    }
    return mapping.get(top_name, "mean_reversion")
