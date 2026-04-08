from rlm.datasets.paths import (
    rel_bars_csv,
    rel_features_csv,
    rel_option_chain_csv,
)


def test_rel_paths_use_upper_symbol() -> None:
    assert rel_bars_csv("qqq") == "data/raw/bars_QQQ.csv"
    assert rel_option_chain_csv("spy") == "data/raw/option_chain_SPY.csv"
    assert rel_features_csv("SPY") == "data/processed/features_SPY.csv"