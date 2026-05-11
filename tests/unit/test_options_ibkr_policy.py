from __future__ import annotations

import pytest

from rlm.execution import options_ibkr_policy as pol


def test_ibkr_option_orders_allowed_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_ALLOW_IBKR_OPTIONS", raising=False)
    assert pol.ibkr_option_orders_allowed() is False


def test_ibkr_option_orders_allowed_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_ALLOW_IBKR_OPTIONS", "1")
    assert pol.ibkr_option_orders_allowed() is True


def test_exit_if_disallowed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_ALLOW_IBKR_OPTIONS", raising=False)
    with pytest.raises(SystemExit) as ei:
        pol.exit_if_ibkr_option_orders_disallowed("test")
    assert int(ei.value.code) == 2


def test_exit_if_allowed_no_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_ALLOW_IBKR_OPTIONS", "1")
    pol.exit_if_ibkr_option_orders_disallowed("test")  # no exception
