from __future__ import annotations

import math

import pandas as pd

from rlm.options.edge import assess_combo_edge, chain_spot_price, enrich_leg_greeks, plan_underlying_close


def test_enrich_leg_greeks_fills_delta() -> None:
    leg = {
        "side": "long",
        "option_type": "call",
        "strike": 500.0,
        "expiry": "2099-12-31",
        "bid": 1.0,
        "ask": 1.2,
        "mid": 1.1,
        "iv": 0.2,
    }
    out = enrich_leg_greeks(leg, spot=500.0)
    assert out.get("delta") is not None
    assert math.isfinite(float(out["delta"]))


def test_assess_combo_edge_rejects_wide_spread(monkeypatch) -> None:
    monkeypatch.setenv("RLM_OPTIONS_MAX_SPREAD_PCT_MID", "0.05")
    legs = [
        {
            "side": "long",
            "option_type": "call",
            "strike": 500.0,
            "expiry": "2099-12-31",
            "bid": 1.0,
            "ask": 2.0,
            "mid": 1.5,
            "iv": 0.2,
            "delta": 0.5,
        }
    ]
    edge = assess_combo_edge(
        legs,
        spot=500.0,
        entry_debit_dollars=150.0,
        regime_direction="bull",
        strategy_name="long_call",
    )
    assert edge["passes_edge_gate"] is False


def test_chain_spot_price_ignores_ticker_underlying_and_uses_fallback() -> None:
    """Massive chains store the ticker in ``underlying`` and have no price column."""
    chain = pd.DataFrame(
        {
            "underlying": ["SPY", "SPY"],
            "strike": [500.0, 505.0],
            "bid": [1.0, 1.1],
            "ask": [1.2, 1.3],
        }
    )
    assert chain_spot_price(chain, fallback=512.4) == 512.4


def test_chain_spot_price_skips_non_numeric_price_columns() -> None:
    chain = pd.DataFrame(
        {
            "underlying_price": [float("nan"), float("nan")],
            "spot": ["", None],
            "underlying": ["QQQ", "QQQ"],
        }
    )
    assert chain_spot_price(chain, fallback=430.0) == 430.0


def test_chain_spot_price_prefers_numeric_underlying_price() -> None:
    chain = pd.DataFrame(
        {
            "underlying_price": [float("nan"), 501.25],
            "underlying": ["SPY", "SPY"],
        }
    )
    assert chain_spot_price(chain, fallback=10.0) == 501.25


def test_plan_underlying_close_reads_pipeline_close() -> None:
    assert plan_underlying_close({"pipeline": {"close": 188.5}, "last_close": 1.0}) == 188.5
    assert plan_underlying_close({}) is None
