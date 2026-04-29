from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def _without_modules(*names: str) -> Iterator[None]:
    originals = {name: sys.modules.get(name) for name in names}
    parent_attrs: dict[str, tuple[object, bool, object | None]] = {}
    for name in names:
        parent_name, _, attr = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            parent_attrs[name] = (parent, hasattr(parent, attr), getattr(parent, attr, None))
    try:
        for name in names:
            sys.modules.pop(name, None)
        yield
    finally:
        for name, module in originals.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name, (parent, had_attr, value) in parent_attrs.items():
            _, _, attr = name.rpartition(".")
            if had_attr:
                setattr(parent, attr, value)
            elif hasattr(parent, attr):
                delattr(parent, attr)


def test_execution_package_reexports_helpers_lazily() -> None:
    with _without_modules("rlm.execution", "rlm.execution.ibkr_combo_orders"):
        import rlm.execution as execution

        assert execution.__name__ == "rlm.execution"
        assert "rlm.execution.ibkr_combo_orders" not in sys.modules

        from rlm.execution import IBKROptionLegSpec, build_spread_exit_thresholds

        assert IBKROptionLegSpec.__name__ == "IBKROptionLegSpec"
        assert callable(build_spread_exit_thresholds)


def test_notify_package_reexports_helpers_lazily() -> None:
    with _without_modules("rlm.notify", "rlm.notify.telegram_rlm"):
        import rlm.notify as notify

        assert notify.__name__ == "rlm.notify"
        assert "rlm.notify.telegram_rlm" not in sys.modules

        from rlm.notify import build_status_brief, notification_cycle

        assert callable(build_status_brief)
        assert callable(notification_cycle)
