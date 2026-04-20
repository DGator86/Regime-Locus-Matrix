from rlm.cli.io import resolve_output_path


def test_resolve_output_default(tmp_path):
    out = resolve_output_path("forecast_features", "SPY", None, str(tmp_path))
    assert out.name == "forecast_features_SPY.csv"
