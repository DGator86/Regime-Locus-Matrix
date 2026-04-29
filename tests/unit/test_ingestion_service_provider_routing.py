from __future__ import annotations

import pandas as pd

from rlm.core.services.ingestion_service import IngestionRequest, IngestionService


class _StubProvider:
    def fetch_bars(self, *, symbol: str, start: str | None, end: str | None, interval: str):
        del symbol, start, end, interval
        from rlm.data.providers.base import ProviderBarsResult

        return ProviderBarsResult(pd.DataFrame([{"timestamp": "2025-01-01", "open": 1.0}]), source="stub")

    def fetch_option_chain(self, *, symbol: str):
        del symbol
        from rlm.data.providers.base import ProviderOptionChainResult

        return ProviderOptionChainResult(pd.DataFrame([{"contract": "x"}]), source="stub")


def test_ingestion_service_routes_provider(monkeypatch, tmp_path):
    monkeypatch.setattr("rlm.core.services.ingestion_service.resolve_provider", lambda _name: _StubProvider())

    req = IngestionRequest(symbol="SPY", source="stub", fetch_options=True, data_root=str(tmp_path), backend="csv")
    result = IngestionService().run(req)

    assert result.provider == "stub"
    assert result.bars_count == 1
    assert result.chain_count == 1
