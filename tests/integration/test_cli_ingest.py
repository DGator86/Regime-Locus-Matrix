from __future__ import annotations

import sys

from rlm.cli import ingest


def test_cli_ingest_invokes_service(monkeypatch, tmp_path, capsys):
    called = {}

    class _Svc:
        def run(self, req):
            called["backend"] = req.backend
            from rlm.core.services.ingestion_service import IngestionResult

            out = tmp_path / "raw" / "bars_SPY.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("timestamp,open\n", encoding="utf-8")
            return IngestionResult(bars_path=out, bars_count=0)

    monkeypatch.setattr(ingest, "IngestionService", lambda: _Svc())
    monkeypatch.setattr(sys, "argv", ["rlm ingest", "--symbol", "SPY", "--backend", "csv"])
    ingest.main()
    stdout = capsys.readouterr().out
    assert "Wrote" in stdout
    assert called["backend"] == "csv"
