from __future__ import annotations

import sys


def test_execution_package_reexports_helpers_lazily() -> None:
    sys.modules.pop("rlm.execution", None)
    sys.modules.pop("rlm.execution.ibkr_combo_orders", None)

    import rlm.execution as execution

    assert "rlm.execution.ibkr_combo_orders" not in sys.modules

    from rlm.execution import IBKROptionLegSpec, build_spread_exit_thresholds

    assert IBKROptionLegSpec.__name__ == "IBKROptionLegSpec"
    assert callable(build_spread_exit_thresholds)


def test_notify_package_reexports_helpers_lazily() -> None:
    sys.modules.pop("rlm.notify", None)
    sys.modules.pop("rlm.notify.telegram_rlm", None)

    import rlm.notify as notify

    assert "rlm.notify.telegram_rlm" not in sys.modules

    from rlm.notify import build_status_brief, notification_cycle

    assert callable(build_status_brief)
    assert callable(notification_cycle)
