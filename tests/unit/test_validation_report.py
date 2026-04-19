from __future__ import annotations

import json
from pathlib import Path

from rlm.training.validation_matrix import ValidationSliceResult
from rlm.training.validation_report import save_validation_report


def test_validation_report_saves_json(tmp_path: Path) -> None:
    results = [
        ValidationSliceResult(
            symbol="SPY",
            start="2026-03-01T00:00:00Z",
            end="2026-03-01T01:00:00Z",
            selected_realized_avg_improvement_vs_pr51=0.01,
            top1_hit_rate_improvement_vs_pr51=0.02,
            regime_flip_rate_improvement_vs_pr51=0.10,
            drawdown_proxy_improvement_vs_pr51=0.05,
        )
    ]
    out = tmp_path / "validation.json"
    save_validation_report(results, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["symbol"] == "SPY"
