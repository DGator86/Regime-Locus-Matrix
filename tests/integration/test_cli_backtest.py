from rlm.data.synthetic import synthetic_bars_demo


def test_synthetic_demo_import():
    df = synthetic_bars_demo(end="2024-01-01", periods=10)
    assert len(df) == 10
