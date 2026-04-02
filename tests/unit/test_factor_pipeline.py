import numpy as np
import pandas as pd

from rlm.factors.pipeline import FactorPipeline


def make_sample_bars(n: int = 150) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    base = np.linspace(5000, 5050, n)

    df = pd.DataFrame(
        {
            "open": base - 1.0,
            "high": base + 3.0,
            "low": base - 3.0,
            "close": base + np.sin(np.arange(n) / 8.0) * 5.0,
            "volume": 100000 + (np.arange(n) % 10) * 5000,
            "vwap": base,
            "anchored_vwap": base - 2.0,
            "buy_volume": 55000 + (np.arange(n) % 7) * 2000,
            "sell_volume": 45000 + (np.arange(n) % 5) * 1500,
            "advancers": 1800 + (np.arange(n) % 20) * 10,
            "decliners": 1400 + (np.arange(n) % 15) * 10,
            "index_return": pd.Series(base).pct_change(10).fillna(0.0).values,
            "vix": 18 + (np.arange(n) % 12) * 0.5,
            "vvix": 90 + (np.arange(n) % 9) * 1.5,
            "bid_ask_spread": 0.5 + (np.arange(n) % 6) * 0.02,
            "order_book_depth": 2000 + (np.arange(n) % 8) * 100,
            "gex": np.sin(np.arange(n) / 10.0),
            "vanna": np.cos(np.arange(n) / 12.0),
            "charm": np.sin(np.arange(n) / 7.0),
            "put_call_skew": 0.02 + (np.arange(n) % 5) * 0.005,
            "iv_rank": 0.45 + (np.arange(n) % 10) * 0.02,
            "term_structure_ratio": 0.95 + (np.arange(n) % 6) * 0.02,
            "dealer_position_proxy": np.sin(np.arange(n) / 15.0) * 0.2,
        },
        index=idx,
    )
    return df


def test_factor_pipeline_outputs_composite_scores() -> None:
    df = make_sample_bars()
    pipeline = FactorPipeline()
    out = pipeline.run(df)

    for col in ["S_D", "S_V", "S_L", "S_G"]:
        assert col in out.columns

    assert out["S_D"].notna().sum() > 0
    assert out["S_V"].notna().sum() > 0
    assert out["S_L"].notna().sum() > 0
    assert out["S_G"].notna().sum() > 0


def test_standardized_factors_are_bounded() -> None:
    df = make_sample_bars()
    pipeline = FactorPipeline()
    out = pipeline.run(df)

    std_cols = [c for c in out.columns if c.startswith("std_")]
    vals = out[std_cols].to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]

    assert finite_vals.size > 0
    assert (finite_vals <= 1.0).all()
    assert (finite_vals >= -1.0).all()
