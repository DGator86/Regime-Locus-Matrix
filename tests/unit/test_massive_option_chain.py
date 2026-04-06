import pandas as pd

from rlm.data.massive_option_chain import (
    collect_option_snapshot_pages,
    massive_option_snapshot_to_normalized_chain,
)


def test_massive_option_snapshot_to_normalized_chain() -> None:
    ts = pd.Timestamp("2024-01-15")
    payload = {
        "status": "OK",
        "results": [
            {
                "details": {
                    "ticker": "O:AAPL240202C00185000",
                    "contract_type": "call",
                    "expiration_date": "2024-02-02",
                    "strike_price": 185.0,
                },
                "last_quote": {"bid": 2.1, "ask": 2.15},
                "greeks": {"delta": 0.5, "gamma": 0.01, "theta": -0.02, "vega": 0.03},
                "implied_volatility": 0.28,
                "open_interest": 1000,
                "day": {"volume": 42},
            }
        ],
    }
    out = massive_option_snapshot_to_normalized_chain(payload, underlying="AAPL", timestamp=ts)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["underlying"] == "AAPL"
    assert row["strike"] == 185.0
    assert row["option_type"] == "call"
    assert abs(row["bid"] - 2.1) < 1e-9
    assert abs(row["ask"] - 2.15) < 1e-9
    assert row["delta"] == 0.5
    assert row["iv"] == 0.28
    assert row["open_interest"] == 1000
    assert row["volume"] == 42


def test_collect_option_snapshot_pages_merges_next_url() -> None:
    class _Fake:
        def __init__(self) -> None:
            self._n = 0

        def option_chain_snapshot(self, u: str, **kw: object) -> dict:
            assert u == "SPY"
            self._n += 1
            return {
                "results": [
                    {
                        "details": {
                            "ticker": "O:SPY240119C00480000",
                            "expiration_date": "2024-01-19",
                            "contract_type": "call",
                            "strike_price": 480,
                        },
                        "last_quote": {"bid": 1, "ask": 1.1},
                    }
                ],
                "next_url": "https://api.massive.com/v3/snapshot/options/SPY?cursor=abc",
            }

        def get_by_url(self, url: str) -> dict:
            assert "cursor=abc" in url
            return {
                "results": [
                    {
                        "details": {
                            "ticker": "O:SPY240119P00480000",
                            "expiration_date": "2024-01-19",
                            "contract_type": "put",
                            "strike_price": 480,
                        },
                        "last_quote": {"bid": 2, "ask": 2.1},
                    }
                ],
            }

    rows = collect_option_snapshot_pages(_Fake(), "SPY")  # type: ignore[arg-type]
    assert len(rows) == 2
