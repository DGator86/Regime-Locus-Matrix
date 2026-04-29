from __future__ import annotations

import importlib
import sys


def _reload_without_submodules(package_name: str, submodule_prefix: str):
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(submodule_prefix):
            sys.modules.pop(module_name, None)
    return importlib.import_module(package_name)


def test_execution_package_exports_public_helpers_lazily() -> None:
    execution = _reload_without_submodules("rlm.execution", "rlm.execution.")

    assert execution.__name__ == "rlm.execution"
    assert "rlm.execution.ibkr_combo_orders" not in sys.modules
    assert "rlm.execution.risk_targets" not in sys.modules

    from rlm.execution import IBKROptionLegSpec, trailing_stop_from_peak

    assert IBKROptionLegSpec.__name__ == "IBKROptionLegSpec"
    assert trailing_stop_from_peak(100.0, 0.2) == 80.0


def test_notify_package_exports_public_helpers_lazily() -> None:
    notify = _reload_without_submodules("rlm.notify", "rlm.notify.")

    assert notify.__name__ == "rlm.notify"
    assert "rlm.notify.telegram_rlm" not in sys.modules

    from rlm.notify import build_status_brief, notification_cycle

    assert callable(build_status_brief)
    assert callable(notification_cycle)
