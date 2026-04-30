from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def _reload_without_submodules(package_name: str, submodule_prefix: str) -> Iterator[object]:
    parent_name, _, child_name = package_name.rpartition(".")
    parent_module = importlib.import_module(parent_name) if parent_name else None
    sentinel = object()
    previous_child = getattr(parent_module, child_name, sentinel) if parent_module is not None else sentinel
    saved = {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name == package_name or module_name.startswith(submodule_prefix)
    }
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(submodule_prefix):
            sys.modules.pop(module_name, None)
    try:
        yield importlib.import_module(package_name)
    finally:
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(submodule_prefix):
                sys.modules.pop(module_name, None)
        sys.modules.update(saved)
        if parent_module is not None:
            if previous_child is sentinel:
                try:
                    delattr(parent_module, child_name)
                except AttributeError:
                    pass
            else:
                setattr(parent_module, child_name, previous_child)


def test_execution_package_exports_public_helpers_lazily() -> None:
    with _reload_without_submodules("rlm.execution", "rlm.execution.") as execution:

        assert execution.__name__ == "rlm.execution"
        assert "rlm.execution.ibkr_combo_orders" not in sys.modules
        assert "rlm.execution.risk_targets" not in sys.modules

        from rlm.execution import IBKROptionLegSpec, trailing_stop_from_peak

        assert IBKROptionLegSpec.__name__ == "IBKROptionLegSpec"
        assert trailing_stop_from_peak(100.0, 0.2) == 80.0


def test_notify_package_exports_public_helpers_lazily() -> None:
    with _reload_without_submodules("rlm.notify", "rlm.notify.") as notify:

        assert notify.__name__ == "rlm.notify"
        assert "rlm.notify.telegram_rlm" not in sys.modules

        from rlm.notify import build_status_brief, notification_cycle

        assert callable(build_status_brief)
        assert callable(notification_cycle)
