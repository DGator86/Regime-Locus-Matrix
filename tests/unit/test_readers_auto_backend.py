import pandas as pd

from rlm.data.readers import load_bars


def test_load_bars_csv_auto(tmp_path):
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3), "close": [1, 2, 3]})
    df.to_csv(raw / "bars_SPY.csv", index=False)

    out = load_bars("SPY", data_root=root, backend="auto")
    assert len(out) == 3
