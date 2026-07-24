from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from rlm.utils.atomic_io import write_json_atomic


def test_write_json_atomic_prevents_truncated_reads(tmp_path: Path) -> None:
    path = tmp_path / "universe_trade_plans.json"
    payload = {
        "generated_at_utc": "2026-07-24T15:00:00Z",
        "symbols_requested": [f"SYM{i}" for i in range(40)],
        "results": [
            {
                "plan_id": f"SYM{i}_20260724_1500",
                "symbol": f"SYM{i}",
                "status": "active" if i < 5 else "skipped",
                "matched_legs": [{"strike": 100 + i, "expiry": "2026-08-21", "right": "C"}] * 2,
                "decision": {"strategy_name": "long_call", "reason": "x" * 120},
                "rank_score": float(i),
            }
            for i in range(60)
        ],
    }
    payload["active_ranked"] = [r for r in payload["results"] if r["status"] == "active"]
    write_json_atomic(path, payload)

    errors: list[str] = []
    ok = {"n": 0}

    def reader() -> None:
        for _ in range(400):
            try:
                json.loads(path.read_text(encoding="utf-8"))
                ok["n"] += 1
            except Exception as exc:  # noqa: BLE001 - collect race failures
                errors.append(f"{type(exc).__name__}: {exc}")

    def writer() -> None:
        for _ in range(80):
            write_json_atomic(path, payload)
            time.sleep(0.001)

    t_read = threading.Thread(target=reader)
    t_write = threading.Thread(target=writer)
    t_read.start()
    t_write.start()
    t_read.join()
    t_write.join()

    assert ok["n"] > 0
    assert errors == [], errors[:5]
    assert json.loads(path.read_text(encoding="utf-8"))["active_ranked"]
