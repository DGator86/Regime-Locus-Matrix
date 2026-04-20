from __future__ import annotations

from rlm.data.providers.massive_provider import MassiveProvider


def test_massive_provider_importable():
    provider = MassiveProvider()
    assert provider.source == "massive"
