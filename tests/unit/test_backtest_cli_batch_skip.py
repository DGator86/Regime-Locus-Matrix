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
