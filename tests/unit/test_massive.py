import json
from http.client import IncompleteRead
from unittest.mock import patch

import pytest

from rlm.data.massive import MassiveClient


def test_massive_get_parses_json() -> None:
    payload = {"status": "OK", "results": []}

    class _Resp:
        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *a: object) -> None:
            pass

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(req: object, timeout: float | None = None) -> _Resp:
        full = getattr(req, "full_url", "") or ""
        assert "apiKey=test-key" in full
        assert "/v3/snapshot/options/SPY" in full
        return _Resp()

    with patch("rlm.data.massive.urlopen", side_effect=fake_urlopen):
        c = MassiveClient(api_key="test-key")
        out = c.option_chain_snapshot("SPY")

    assert out == payload


def test_massive_client_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setattr("rlm.data.massive.load_dotenv", lambda *a, **k: None)
    with pytest.raises(ValueError, match="MASSIVE_API_KEY"):
        MassiveClient()


def test_massive_retries_incomplete_read(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"status": "OK", "results": []}

    class _Resp:
        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *a: object) -> None:
            pass

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    calls = {"n": 0}

    def fake_urlopen(req: object, timeout: float | None = None) -> _Resp:
        calls["n"] += 1
        if calls["n"] == 1:
            raise IncompleteRead(b"x", 2)
        return _Resp()

    monkeypatch.setenv("RLM_MASSIVE_RETRIES", "2")
    with patch("rlm.data.massive.urlopen", side_effect=fake_urlopen):
        c = MassiveClient(api_key="test-key")
        out = c.option_chain_snapshot("SPY")

    assert out == payload
    assert calls["n"] == 2
