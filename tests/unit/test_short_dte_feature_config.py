from __future__ import annotations

from rlm.features.factors.config import feature_config_for_pipeline


def test_short_dte_overlay_includes_candles() -> None:
    cfg = feature_config_for_pipeline(short_dte=True)
    enabled = cfg.get("enabled_factors") or {}
    assert "candle_bullish_reversal" in enabled.get("direction", [])
