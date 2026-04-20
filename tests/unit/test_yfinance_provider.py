from __future__ import annotations

import pandas as pd

from rlm.data.providers.yfinance_provider import YFinanceProvider


def test_yfinance_provider_bars(monkeypatch):
    class _Ticker:
        options = []

        def history(self, **_kwargs):
            idx = pd.date_range("2025-01-01", periods=2, freq="D")
            return pd.DataFrame({"Open": [1, 2], "Close": [2, 3]}, index=idx)

    monkeypatch.setattr("rlm.data.providers.yfinance_provider.yf.Ticker", lambda _s: _Ticker())
    out = YFinanceProvider().fetch_bars(symbol="SPY", start=None, end=None, interval="1d")
    assert len(out.bars_df) == 2
    assert "open" in out.bars_df.columns
