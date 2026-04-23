"""ChallengeTracker — persistence layer for challenge state.

State is written as JSON to ``data/challenge/state.json``.
Closed trades are appended to ``data/challenge/trade_log.csv``.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.state import ChallengeState, ChallengeTradeRecord


class ChallengeTracker:
    """Load / save challenge state and append trade records."""

    def __init__(self, data_root: str | None = None) -> None:
        base = Path(data_root).expanduser() if data_root else Path("data")
        self._dir = base / "challenge"
        self._state_path = self._dir / "state.json"
        self._trade_log_path = self._dir / "trade_log.csv"

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return self._state_path.exists()

    def load(self) -> ChallengeState:
        if not self._state_path.exists():
            raise FileNotFoundError(
                f"No challenge state found at {self._state_path}. "
                "Run `rlm challenge --reset` to start a new challenge."
            )
        raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        return ChallengeState.from_dict(raw)

    def save(self, state: ChallengeState) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(state.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    def reset(self, cfg: ChallengeConfig) -> ChallengeState:
        self._dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(tz=timezone.utc).isoformat()
        state = ChallengeState.fresh(cfg, now)
        self.save(state)
        # Archive old trade log rather than delete
        if self._trade_log_path.exists():
            archive = self._trade_log_path.with_suffix(f".{now[:10]}.csv")
            self._trade_log_path.rename(archive)
        return state

    # ------------------------------------------------------------------
    # Trade log
    # ------------------------------------------------------------------

    def append_trade(self, record: ChallengeTradeRecord) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        write_header = not self._trade_log_path.exists()
        row = record.to_dict()
        with self._trade_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def trade_log_path(self) -> Path:
        return self._trade_log_path

    def state_path(self) -> Path:
        return self._state_path
