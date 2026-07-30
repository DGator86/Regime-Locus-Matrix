from __future__ import annotations

import math

from rlm.options.edge import assess_combo_edge, enrich_leg_greeks


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
    assert "_iv_solved_from_mid" not in out


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


def test_mid_implied_iv_does_not_false_reject_debit() -> None:
    """Solving IV from mid makes fair≈mid; that must not trip min_edge."""
    legs = [
        {
            "side": "long",
            "option_type": "call",
            "strike": 100.0,
            "expiry": "2026-08-15",
            "bid": 4.9,
            "ask": 5.1,
            "mid": 5.0,
        },
        {
            "side": "short",
            "option_type": "call",
            "strike": 105.0,
            "expiry": "2026-08-15",
            "bid": 2.9,
            "ask": 3.1,
            "mid": 3.0,
        },
    ]
    edge = assess_combo_edge(
        legs,
        spot=100.0,
        entry_debit_dollars=200.0,
        regime_direction="bull",
        strategy_name="bull_call_spread",
    )
    assert math.isnan(edge["buyer_edge_pct"])
    assert math.isnan(edge["fair_mark_dollars"])
    assert edge["passes_edge_gate"] is True
    assert all("_iv_solved_from_mid" not in leg for leg in edge["matched_legs_enriched"])


def test_missing_fair_is_unknown_not_worthless() -> None:
    """Expired / unsolvable IV must not score buyer_edge as -100%."""
    legs = [
        {
            "side": "long",
            "option_type": "call",
            "strike": 100.0,
            "expiry": "2020-01-01",
            "bid": 1.05,
            "ask": 1.15,
            "mid": 1.1,
            "delta": 0.5,
        }
    ]
    edge = assess_combo_edge(
        legs,
        spot=100.0,
        entry_debit_dollars=110.0,
        regime_direction="bull",
        strategy_name="long_call",
    )
    assert math.isnan(edge["buyer_edge_pct"])
    assert math.isnan(edge["fair_mark_dollars"])
    assert edge["passes_edge_gate"] is True


def test_chain_iv_still_rejects_overpriced_debit() -> None:
    """Independent chain IV below mid-implied → negative edge → reject."""
    legs = [
        {
            "side": "long",
            "option_type": "call",
            "strike": 100.0,
            "expiry": "2026-08-15",
            "bid": 4.9,
            "ask": 5.1,
            "mid": 5.0,
            "iv": 0.30,
            "delta": 0.55,
        },
        {
            "side": "short",
            "option_type": "call",
            "strike": 105.0,
            "expiry": "2026-08-15",
            "bid": 2.9,
            "ask": 3.1,
            "mid": 3.0,
            "iv": 0.30,
            "delta": 0.35,
        },
    ]
    edge = assess_combo_edge(
        legs,
        spot=100.0,
        entry_debit_dollars=200.0,
        regime_direction="bull",
        strategy_name="bull_call_spread",
    )
    assert edge["buyer_edge_pct"] < 0
    assert edge["passes_edge_gate"] is False
    assert "debit_overpriced" in str(edge["edge_skip_reason"])


def test_credit_gate_accepts_rich_credit_rejects_cheap() -> None:
    """Credit sellers need positive buyer_edge (richer credit than model)."""
    rich_legs = [
        {
            "side": "short",
            "option_type": "call",
            "strike": 105.0,
            "expiry": "2026-08-15",
            "bid": 4.95,
            "ask": 5.05,
            "mid": 5.0,
            "iv": 0.30,
            "delta": -0.35,
        },
        {
            "side": "long",
            "option_type": "call",
            "strike": 110.0,
            "expiry": "2026-08-15",
            "bid": 2.95,
            "ask": 3.05,
            "mid": 3.0,
            "iv": 0.30,
            "delta": -0.22,
        },
    ]
    rich = assess_combo_edge(
        rich_legs,
        spot=100.0,
        entry_debit_dollars=-200.0,
        regime_direction="",
        strategy_name="bear_call_credit_spread",
    )
    assert rich["buyer_edge_pct"] >= 0.01
    assert rich["passes_edge_gate"] is True

    # Market credit too small vs high chain-IV fair credit → underpriced for seller.
    cheap_legs = [
        {
            "side": "short",
            "option_type": "call",
            "strike": 105.0,
            "expiry": "2026-08-15",
            "bid": 1.95,
            "ask": 2.05,
            "mid": 2.0,
            "iv": 0.40,
            "delta": -0.30,
        },
        {
            "side": "long",
            "option_type": "call",
            "strike": 110.0,
            "expiry": "2026-08-15",
            "bid": 1.45,
            "ask": 1.55,
            "mid": 1.5,
            "iv": 0.40,
            "delta": -0.18,
        },
    ]
    cheap = assess_combo_edge(
        cheap_legs,
        spot=100.0,
        entry_debit_dollars=-50.0,
        regime_direction="",
        strategy_name="bear_call_credit_spread",
    )
    assert cheap["buyer_edge_pct"] < 0.01
    assert cheap["passes_edge_gate"] is False
    assert "credit_underpriced" in str(cheap["edge_skip_reason"])
