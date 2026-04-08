import pandas as pd

from rlm.backtest.revalue import has_full_reprice, reprice_matched_legs


def test_reprice_matched_legs_updates_values() -> None:
    matched_legs = [
        {
            "side": "long",
            "option_type": "call",
            "strike": 5000.0,
            "expiry": "2025-02-14",
            "bid": 9.0,
            "ask": 9.5,
            "mid": 9.25,
        },
        {
            "side": "short",
            "option_type": "call",
            "strike": 5050.0,
            "expiry": "2025-02-14",
            "bid": 7.0,
            "ask": 7.4,
            "mid": 7.2,
        },
    ]

    snapshot = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-11 10:00:00"),
                "underlying": "SPY",
                "expiry": pd.Timestamp("2025-02-14"),
                "option_type": "call",
                "strike": 5000.0,
                "bid": 10.0,
                "ask": 10.4,
                "mid": 10.2,
                "spread_pct_mid": 0.039,
            },
            {
                "timestamp": pd.Timestamp("2025-01-11 10:00:00"),
                "underlying": "SPY",
                "expiry": pd.Timestamp("2025-02-14"),
                "option_type": "call",
                "strike": 5050.0,
                "bid": 8.0,
                "ask": 8.3,
                "mid": 8.15,
                "spread_pct_mid": 0.037,
            },
        ]
    )

    repriced = reprice_matched_legs(matched_legs=matched_legs, chain_snapshot=snapshot)
    assert has_full_reprice(matched_legs, repriced)
    assert len(repriced) == 2
    assert repriced[0].mark_value > 0