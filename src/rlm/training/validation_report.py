from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rlm.training.validation_matrix import ValidationSliceResult


def save_validation_report(results: list[ValidationSliceResult], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
