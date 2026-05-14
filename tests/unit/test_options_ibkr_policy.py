from __future__ import annotations

import pytest

from rlm.execution import options_ibkr_policy as pol


def test_exit_ibkr_option_combo_blocked_always_exits() -> None:
    with pytest.raises(SystemExit) as ei:
        pol.exit_ibkr_option_combo_blocked("test context")
    assert int(ei.value.code) == 2
