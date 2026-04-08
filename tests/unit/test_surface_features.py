import numpy as np
import pandas as pd

from rlm.datasets.bars_enrichment import enrich_bars_from_option_chain


def _make_bars() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1_000_000] * 4,
            "vwap": [100.0, 101.0, 102.0, 103.0],
        },
        index=idx,
    )


def _make_chain(bars: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp | str]] = []
    expiries = [pd.Timestamp("2025-01-20"), pd.Timestamp("2025-02-20")]
    for ts, row in bars.iterrows():
        spot = float(row["close"])
        for expiry in expiries:
            tau_days = max((expiry.normalize() - pd.Timestamp(ts).normalize()).days, 1)
            tau = tau_days / 365.0
            forward = spot * 1.001
            for option_type in ("call", "put"):
                for strike in [spot - 10.0, spot - 5.0, spot, spot + 5.0, spot + 10.0]:
                    rel = abs(np.log(float(strike) / forward))
                    iv = 0.18 + 0.04 * rel + (0.01 if option_type == "put" else 0.0)
                    mid = 1.0 + rel * 25.0 + tau * 5.0
                    rows.append(
                        {
                            "timestamp": ts,
                            "underlying": "SPY",
                            "expiry": expiry,
                            "option_type": option_type,
                            "strike": float(strike),
                            "bid": mid - 0.1,
                            "ask": mid + 0.1,
                            "iv": iv,
                            "open_interest": 5_000,
                            "volume": 500,
                        }
                    )
    return pd.DataFrame(rows)


def test_enrich_bars_adds_surface_features() -> None:
    bars = _make_bars()
    chain = _make_chain(bars)

    out = enrich_bars_from_option_chain(bars, chain, underlying="SPY")

    for col in [
        "surface_atm_forward_iv",
        "surface_skew",
        "surface_convexity",
        "surface_term_slope",
    ]:
        assert col in out.columns
        assert out[col].notna().any()
