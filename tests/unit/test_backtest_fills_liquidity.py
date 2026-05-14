from rlm.backtest.fills import FillConfig, entry_fill_price


def test_entry_fill_price_applies_liquidity_impact_when_size_exceeds_depth() -> None:
    cfg = FillConfig(liquidity_impact_factor=1.0)
    small = entry_fill_price(side="long", bid=1.0, ask=1.2, config=cfg, quantity=1, quote_size=10)
    large = entry_fill_price(side="long", bid=1.0, ask=1.2, config=cfg, quantity=50, quote_size=10)
    assert large > small
