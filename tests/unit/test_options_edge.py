from __future__ import annotations

import math

import pandas as pd

from rlm.options.edge import assess_combo_edge, chain_spot_price, enrich_leg_greeks


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


def test_chain_spot_price_falls_back_when_underlying_is_symbol() -> None:
    chain = pd.DataFrame(
        [
            {
                "underlying": "SPY",
                "bid": 1.0,
                "ask": 1.2,
            }
        ]
    )

    assert chain_spot_price(chain, fallback=505.25) == 505.25


def test_chain_spot_price_uses_numeric_spot_column() -> None:
    chain = pd.DataFrame(
        [
            {
                "underlying": "SPY",
                "underlying_price": "502.5",
            }
        ]
    )

    assert chain_spot_price(chain, fallback=505.25) == 502.5
