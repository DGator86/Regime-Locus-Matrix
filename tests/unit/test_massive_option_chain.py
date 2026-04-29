import time
from threading import Lock

import pandas as pd

from rlm.data.massive_option_chain import (
    clear_massive_option_chain_ram_cache,
    collect_option_snapshot_pages,
    massive_option_chain_from_client,
    massive_option_chains_from_client,
    massive_option_snapshot_to_normalized_chain,
)
from scripts.monitor_active_trade_plans import (
    _build_incremental_snapshot_params as monitor_incremental_params,
)
from scripts.run_universe_options_pipeline import (
    _build_incremental_snapshot_params as universe_incremental_params,
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


def test_massive_option_chain_from_client_uses_ram_cache() -> None:
    clear_massive_option_chain_ram_cache()

    class _Fake:
        def __init__(self) -> None:
            self.calls = 0

        def option_chain_snapshot(self, u: str, **kw: object) -> dict:
            assert u == "SPY"
            assert kw["limit"] == 250
            self.calls += 1
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
                ]
            }

        def get_by_url(self, url: str) -> dict:
            raise AssertionError(f"Unexpected pagination fetch: {url}")

    fake = _Fake()
    first = massive_option_chain_from_client(
        fake,  # type: ignore[arg-type]
        "SPY",
        cache_ttl_s=30.0,
        use_ram_cache=True,
        limit=250,
    )
    second = massive_option_chain_from_client(
        fake,  # type: ignore[arg-type]
        "SPY",
        cache_ttl_s=30.0,
        use_ram_cache=True,
        limit=250,
    )

    assert fake.calls == 1
    pd.testing.assert_frame_equal(first, second)
    clear_massive_option_chain_ram_cache()


def test_massive_option_chains_from_client_fetches_symbols_in_parallel() -> None:
    clear_massive_option_chain_ram_cache()
    lock = Lock()
    active_calls = 0
    max_active_calls = 0
    params_seen: dict[str, dict[str, object]] = {}

    class _Fake:
        def option_chain_snapshot(self, u: str, **kw: object) -> dict:
            nonlocal active_calls, max_active_calls
            with lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
                params_seen[u] = dict(kw)
            time.sleep(0.05)
            with lock:
                active_calls -= 1
            return {
                "results": [
                    {
                        "details": {
                            "ticker": f"O:{u}240119C00480000",
                            "expiration_date": "2024-01-19",
                            "contract_type": "call",
                            "strike_price": 480,
                        },
                        "last_quote": {"bid": 1, "ask": 1.1},
                    }
                ]
            }

        def get_by_url(self, url: str) -> dict:
            raise AssertionError(f"Unexpected pagination fetch: {url}")

    batch = massive_option_chains_from_client(
        _Fake(),  # type: ignore[arg-type]
        ["SPY", "QQQ", "AAPL"],
        max_workers=3,
        limit=250,
        per_symbol_params={
            "SPY": {"expiration_date.gte": "2024-01-19"},
            "QQQ": {"contract_type": "call"},
        },
    )

    assert not batch.errors
    assert set(batch.chains) == {"SPY", "QQQ", "AAPL"}
    assert max_active_calls >= 2
    assert params_seen["SPY"]["expiration_date.gte"] == "2024-01-19"
    assert params_seen["QQQ"]["contract_type"] == "call"


def test_incremental_snapshot_params_narrow_requests_for_monitor_and_universe() -> None:
    monitor_params = monitor_incremental_params(
        [
            {
                "matched_legs": [
                    {"expiry": "2024-02-16", "strike": 479.0, "option_type": "call"},
                    {"expiry": "2024-02-16", "strike": 485.0, "option_type": "call"},
                ]
            }
        ],
        massive_limit=250,
    )
    assert monitor_params["expiration_date.gte"] == "2024-02-16"
    assert monitor_params["expiration_date.lte"] == "2024-02-16"
    assert monitor_params["strike_price.gte"] == 479.0
    assert monitor_params["strike_price.lte"] == 485.0
    assert monitor_params["contract_type"] == "call"

    from rlm.types.options import OptionLeg, TradeCandidate, TradeDecision

    universe_params = universe_incremental_params(
        pd.Timestamp("2024-01-19"),
        TradeDecision(
            action="enter",
            candidate=TradeCandidate(
                strategy_name="bull_call_spread",
                regime_key="trend_up",
                rationale="test",
                target_dte_min=28,
                target_dte_max=35,
                target_profit_pct=0.25,
                max_risk_pct=0.02,
            ),
            legs=[
                OptionLeg(side="long", option_type="call", strike=480.0),
                OptionLeg(side="short", option_type="call", strike=485.0),
            ],
        ),
        massive_limit=250,
        strike_increment=2.5,
    )
    assert universe_params["expiration_date.gte"] == "2024-02-16"
    assert universe_params["expiration_date.lte"] == "2024-02-23"
    assert universe_params["strike_price.gte"] == 477.5
    assert universe_params["strike_price.lte"] == 487.5
    assert universe_params["contract_type"] == "call"
