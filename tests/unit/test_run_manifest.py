from rlm.core.run_manifest import RunManifest, write_run_manifest


def test_write_run_manifest(tmp_path):
    m = RunManifest(
        run_id="forecast-20260101T000000Z-abc123",
        command="forecast",
        symbol="SPY",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        backend="auto",
        profile=None,
        config_summary={},
        input_paths={},
        output_paths={},
        metrics={},
    )
    out = write_run_manifest(m, data_root=tmp_path)
    assert out.is_file()
