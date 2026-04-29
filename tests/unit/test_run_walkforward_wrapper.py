import runpy
import sys
import types


def _run_wrapper(monkeypatch, argv: list[str]) -> list[str]:
    captured: list[str] = []
    backtest_mod = types.ModuleType("rlm.cli.backtest")

    def fake_main() -> None:
        captured.extend(sys.argv)

    backtest_mod.main = fake_main
    monkeypatch.setitem(sys.modules, "rlm.cli.backtest", backtest_mod)
    monkeypatch.setattr(sys, "argv", argv)

    runpy.run_path("scripts/run_walkforward.py", run_name="__main__")
    return captured


def test_walkforward_wrapper_defaults_to_universe(monkeypatch) -> None:
    argv = _run_wrapper(monkeypatch, ["run_walkforward.py", "--no-vix"])

    assert argv[1:] == ["--walkforward", "--universe", "--no-vix"]


def test_walkforward_wrapper_respects_explicit_symbol(monkeypatch) -> None:
    argv = _run_wrapper(monkeypatch, ["run_walkforward.py", "--symbol", "SPY", "--no-vix"])

    assert argv[1:] == ["--walkforward", "--symbol", "SPY", "--no-vix"]
