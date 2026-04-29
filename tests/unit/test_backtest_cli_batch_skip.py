import argparse

import pytest

from rlm.cli import backtest


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        synthetic=False,
        bars=None,
        chain=None,
        data_root=None,
        backend="csv",
    )


def test_load_symbol_data_skips_missing_bars_in_batch(monkeypatch, capsys):
    def _missing_bars(*args, **kwargs):
        raise FileNotFoundError("missing bars")

    monkeypatch.setattr(backtest, "load_bars", _missing_bars)

    assert backtest._load_symbol_data("AAPL", _args(), batch=True) is None
    assert "AAPL: skipping - missing bars" in capsys.readouterr().out


def test_load_symbol_data_raises_missing_bars_for_single_symbol(monkeypatch):
    def _missing_bars(*args, **kwargs):
        raise FileNotFoundError("missing bars")

    monkeypatch.setattr(backtest, "load_bars", _missing_bars)

    with pytest.raises(FileNotFoundError, match="missing bars"):
        backtest._load_symbol_data("AAPL", _args(), batch=False)


def test_run_symbol_skips_runtime_error_in_batch(monkeypatch, capsys, tmp_path):
    args = _args()
    args.walkforward = False
    args.initial_capital = 100_000.0
    args.profile = None
    args.data_root = str(tmp_path)

    class BrokenService:
        def run(self, req):
            raise TypeError("bad feature")

    monkeypatch.setattr(backtest, "_load_symbol_data", lambda *a, **k: ("bars", None))
    monkeypatch.setattr(backtest, "build_pipeline_config", lambda *a, **k: object())

    assert (
        backtest._run_symbol(
            "SPY",
            args,
            svc=BrokenService(),
            out_dir=tmp_path,
            symbols=["SPY", "QQQ"],
        )
        is None
    )
    assert "SPY: ERROR - bad feature" in capsys.readouterr().out


def test_run_symbol_raises_runtime_error_for_single_symbol(monkeypatch, tmp_path):
    args = _args()
    args.walkforward = False
    args.initial_capital = 100_000.0
    args.profile = None
    args.data_root = str(tmp_path)

    class BrokenService:
        def run(self, req):
            raise TypeError("bad feature")

    monkeypatch.setattr(backtest, "_load_symbol_data", lambda *a, **k: ("bars", None))
    monkeypatch.setattr(backtest, "build_pipeline_config", lambda *a, **k: object())

    with pytest.raises(TypeError, match="bad feature"):
        backtest._run_symbol(
            "SPY",
            args,
            svc=BrokenService(),
            out_dir=tmp_path,
            symbols=["SPY"],
        )
