"""Unit tests for active universe payload parsing."""

from __future__ import annotations

from rlm.universe.active_plans import active_plan_ids, iter_active_trade_plan_rows


def test_iter_active_prefers_ranked_first_and_deduplicates_plan_id() -> None:
    raw = {
        "active_ranked": [
            {"status": "active", "plan_id": "same", "symbol": "A", "score": 1},
        ],
        "results": [
            {"status": "active", "plan_id": "same", "symbol": "B", "score": 2},
            {"status": "active", "plan_id": "only_res", "symbol": "C"},
        ],
    }
    rows = iter_active_trade_plan_rows(raw)
    assert [r["symbol"] for r in rows] == ["A", "C"]
    assert active_plan_ids(raw) == {"same", "only_res"}


def test_inactive_and_missing_plan_id_skipped() -> None:
    raw = {
        "active_ranked": [
            {"status": "inactive", "plan_id": "bad", "symbol": "X"},
            {"status": "active", "symbol": "Y"},
            {"status": "active", "plan_id": "ok", "symbol": "QQQ"},
        ],
        "results": [],
    }
    assert active_plan_ids(raw) == {"ok"}


def test_union_ranked_then_results_when_both_have_distinct_actives() -> None:
    raw = {
        "active_ranked": [{"status": "active", "plan_id": "r1", "symbol": "SPY"}],
        "results": [{"status": "active", "plan_id": "s1", "symbol": "IWM"}],
    }
    assert active_plan_ids(raw) == {"r1", "s1"}
