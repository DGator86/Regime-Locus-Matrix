import pytest

from rlm.execution.ibkr_combo_orders import (
    assert_paper_or_live_acknowledged,
    build_ibkr_algo_fields,
    expiry_iso_to_ib,
    option_type_to_ib_right,
    roee_side_to_ib_action,
)


def test_expiry_iso_to_ib() -> None:
    assert expiry_iso_to_ib("2026-04-18") == "20260418"
    assert expiry_iso_to_ib("20260418") == "20260418"


def test_roee_side_to_ib_action() -> None:
    assert roee_side_to_ib_action("long") == "BUY"
    assert roee_side_to_ib_action("SHORT") == "SELL"


def test_option_type_to_ib_right() -> None:
    assert option_type_to_ib_right("call") == "C"
    assert option_type_to_ib_right("PUT") == "P"


def test_live_port_requires_ack() -> None:
    with pytest.raises(ValueError, match="live"):
        assert_paper_or_live_acknowledged(7496, acknowledge_live=False)
    assert_paper_or_live_acknowledged(7496, acknowledge_live=True) is None
    assert_paper_or_live_acknowledged(7497, acknowledge_live=False) is None


def test_assert_paper_trading_port() -> None:
    from rlm.execution.ibkr_combo_orders import (
        assert_paper_trading_port,
        legs_from_ibkr_combo_spec,
        reverse_legs_for_close,
    )

    assert_paper_trading_port(7497)
    assert_paper_trading_port(4002)
    with pytest.raises(ValueError, match="Automated paper"):
        assert_paper_trading_port(7496)

    spec = {
        "underlying": "SPY",
        "quantity": 1,
        "limit_price": 1.5,
        "legs": [
            {"side": "long", "option_type": "call", "strike": 500.0, "expiry": "2026-06-20"},
            {"side": "short", "option_type": "call", "strike": 510.0, "expiry": "2026-06-20"},
        ],
    }
    legs = legs_from_ibkr_combo_spec(spec)
    assert len(legs) == 2
    assert legs[0][1] == "BUY" and legs[1][1] == "SELL"
    rev = reverse_legs_for_close(legs)
    assert rev[0][1] == "SELL" and rev[1][1] == "BUY"


def test_build_ibkr_algo_fields() -> None:
    name, params = build_ibkr_algo_fields(strategy="NONE")
    assert name is None
    assert params == []

    name, params = build_ibkr_algo_fields(strategy="ADAPTIVE", adaptive_priority="Patient")
    assert name == "Adaptive"
    assert ("adaptivePriority", "Patient") in params

    name, params = build_ibkr_algo_fields(
        strategy="VWAP",
        vwap_max_pct_vol=0.2,
        vwap_start_time="09:35:00 US/Eastern",
        vwap_end_time="15:45:00 US/Eastern",
    )
    assert name == "Vwap"
    assert ("maxPctVol", "0.2000") in params
    assert ("startTime", "09:35:00 US/Eastern") in params
    assert ("endTime", "15:45:00 US/Eastern") in params
