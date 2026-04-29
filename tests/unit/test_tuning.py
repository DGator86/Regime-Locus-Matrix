from rlm.features.optimization.tuning import objective_value


def test_objective_respects_min_trades() -> None:
    s = {"num_trades": 3, "sharpe": 2.0, "max_drawdown": -0.1, "total_return_pct": 0.05}
    assert objective_value(s, "sharpe", min_trades=10) == float("-inf")
    assert objective_value(s, "sharpe", min_trades=2) == 2.0


def test_calmar() -> None:
    s = {"num_trades": 20, "total_return_pct": 0.2, "max_drawdown": -0.1}
    assert abs(objective_value(s, "calmar", min_trades=5) - 2.0) < 1e-9
