"""yfinance-backed market data provider."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from rlm.data.providers.base import ProviderBarsResult, ProviderOptionChainResult


class YFinanceProvider:
    source = "yfinance"

    def fetch_bars(
        self,
        *,
        symbol: str,
        start: str | None,
        end: str | None,
        interval: str,
    ) -> ProviderBarsResult:
        ticker = yf.Ticker(symbol)
        bars = ticker.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
        )
        if bars.empty:
            raise RuntimeError(f"yfinance returned no bars for {symbol}")

        bars.index.name = "timestamp"
        bars.columns = [str(c).lower() for c in bars.columns]
        bars = bars.reset_index()
        return ProviderBarsResult(
            bars_df=bars,
            source=self.source,
            metadata={"interval": interval, "rows": len(bars)},
        )

    def fetch_option_chain(self, *, symbol: str) -> ProviderOptionChainResult:
        ticker = yf.Ticker(symbol)
        frames: list[pd.DataFrame] = []

        for expiry in ticker.options or []:
            chain = ticker.option_chain(expiry)
            for side_name, frame in (("call", chain.calls), ("put", chain.puts)):
                if frame is None or frame.empty:
                    continue
                work = frame.copy()
                work["option_type"] = side_name
                work["expiry"] = pd.to_datetime(expiry)
                frames.append(work)

        if not frames:
            return ProviderOptionChainResult(
                chain_df=None,
                source=self.source,
                metadata={"expiries": 0, "rows": 0},
            )

        chain_df = pd.concat(frames, ignore_index=True)
        return ProviderOptionChainResult(
            chain_df=chain_df,
            source=self.source,
            metadata={"expiries": len(ticker.options or []), "rows": len(chain_df)},
        )
