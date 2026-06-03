from __future__ import annotations

from pathlib import Path

from rlm.trading.tracks import TRACK_LARGE_OPTIONS, TRACK_SPY_DAYTRADE, load_tracks


def test_load_three_tracks() -> None:
    tracks = load_tracks()
    assert TRACK_LARGE_OPTIONS in tracks
    assert TRACK_SPY_DAYTRADE in tracks
    assert tracks[TRACK_LARGE_OPTIONS].short_dte is False
    assert tracks[TRACK_SPY_DAYTRADE].symbol == "SPY"


def test_track_health_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RLM_ROOT", str(tmp_path))
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/challenge").mkdir(parents=True)
    (tmp_path / "data/processed/options_large_account_trade_log.csv").write_text("x", encoding="utf-8")
    from rlm.trading.tracks import track_health

    h = track_health(tmp_path)
    assert h["large_options"]["log_exists"] is True
