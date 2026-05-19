from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from rlm.data import alphavantage_quotes as av


def test_parse_global_quote() -> None:
    payload = {"Global Quote": {"05. price": "123.45"}}
    assert av._parse_global_quote(payload) == 123.45


def test_parse_rate_limit_note() -> None:
    assert av._parse_global_quote({"Note": "rate limit"}) is None


def test_fetch_cached_uses_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RLM_ALPHAVANTAGE_CACHE_SEC", "3600")
    cache_file = tmp_path / "data" / "processed" / "alphavantage_quote_cache.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps({"SPY": {"price": 500.0, "mono": 1.0}}),
        encoding="utf-8",
    )

    with patch.object(av, "fetch_global_quote") as mock_live:
        out = av.fetch_cached_global_quote(tmp_path, "SPY", now_mono=100.0)
        assert out == 500.0
        mock_live.assert_not_called()


def test_fetch_global_quote_mocked(monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")

    class _Resp:
        def read(self) -> bytes:
            return json.dumps({"Global Quote": {"05. price": "99.5"}}).encode()

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    with patch("urllib.request.urlopen", return_value=_Resp()):
        assert av.fetch_global_quote("spy") == 99.5
