import pandas as pd

from rlm.data.massive_stocks import (
    aggregate_trade_flow_to_bars,
    massive_aggs_payload_to_bars_df,
    massive_trades_payload_to_dataframe,
    trades_tick_rule_buy_sell,
)


def test_massive_aggs_payload_to_bars_df() -> None:
    payload = {
        "status": "OK",
        "results": [
            {"o": 10, "h": 11, "l": 9, "c": 10.5, "v": 1000, "vw": 10.2, "t": 1704110400000},
            {"o": 10.5, "h": 12, "l": 10, "c": 11, "v": 2000, "vw": 10.8, "t": 1704114000000},
        ],
    }
    df = massive_aggs_payload_to_bars_df(payload)
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
    assert df.iloc[0]["volume"] == 1000
    assert abs(df.iloc[0]["vwap"] - 10.2) < 1e-9


def test_massive_trades_payload_to_dataframe() -> None:
    payload = {
        "results": [
            {
                "sip_timestamp": 1517562000015577000,
                "price": 100.0,
                "size": 10,
                "exchange": 11,
                "conditions": [12],
            },
            {
                "sip_timestamp": 1517562000015578000,
                "price": 101.0,
                "size": 20,
                "exchange": 11,
                "conditions": [12],
            },
        ]
    }
    df = massive_trades_payload_to_dataframe(payload)
    assert len(df) == 2
    assert "timestamp" in df.columns
    assert df.iloc[1]["price"] == 101.0


def test_tick_rule_and_aggregate_flow() -> None:
    t0 = pd.Timestamp("2024-01-02 14:30:00")
    t1 = pd.Timestamp("2024-01-02 14:31:00")
    bars = pd.DataFrame(
        {
            "timestamp": [t0, t1],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [100, 200],
            "vwap": [100.25, 101.25],
        }
    )
    # Trades inside first minute and second minute
    ts0 = t0 + pd.Timedelta(seconds=10)
    ts1 = t0 + pd.Timedelta(seconds=20)
    ts2 = t1 + pd.Timedelta(seconds=5)
    trades = pd.DataFrame(
        {
            "sip_timestamp": [ts0.value, ts1.value, ts2.value],
            "price": [100.0, 101.0, 100.0],
            "size": [10.0, 20.0, 5.0],
            "timestamp": [ts0, ts1, ts2],
        }
    )
    tagged = trades_tick_rule_buy_sell(trades)
    out = aggregate_trade_flow_to_bars(
        bars,
        tagged,
        bar_time_col="timestamp",
        bar_duration=pd.Timedelta(minutes=1),
    )
    assert out.iloc[0]["buy_volume"] == 30.0  # both upticks
    assert out.iloc[0]["sell_volume"] == 0.0
    assert out.iloc[1]["sell_volume"] == 5.0  # downtick vs prior 101
    assert out.iloc[1]["buy_volume"] == 0.0
