from rlm.data.liquidity_universe import (
    CORE_LIQUID_ETFS,
    LIQUID_STOCK_UNIVERSE_10,
    LIQUID_UNIVERSE,
    MAGNIFICENT_SEVEN,
)


def test_magnificent_seven_count_and_unique() -> None:
    assert len(MAGNIFICENT_SEVEN) == 7
    assert len(set(MAGNIFICENT_SEVEN)) == 7


def test_ten_stocks_universe_unchanged() -> None:
    assert len(LIQUID_STOCK_UNIVERSE_10) == 10
    assert set(CORE_LIQUID_ETFS) == {"SPY", "QQQ"}


def test_liquid_universe_is_mag7_plus_core_etfs() -> None:
    assert len(LIQUID_UNIVERSE) == 9
    assert LIQUID_UNIVERSE[:7] == MAGNIFICENT_SEVEN
    assert LIQUID_UNIVERSE[-2:] == ("SPY", "QQQ")
