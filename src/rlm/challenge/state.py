"""Local JSON state persistence for the challenge engine."""

from __future__ import annotations

import json
from pathlib import Path

from rlm.challenge.models import ChallengeAccountState, PDTTracker

_DEFAULT_DATA_DIR = Path("data") / "challenge"


class ChallengeStateManager:
    """Load and persist ChallengeAccountState + PDTTracker to local JSON."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self._dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._state_path = self._dir / "state.json"
        self._pdt_path = self._dir / "pdt_tracker.json"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[ChallengeAccountState, PDTTracker]:
        """Return (state, pdt).  Creates defaults if files do not exist."""
        state = self._load_state()
        pdt = self._load_pdt()
        return state, pdt

    def save(self, state: ChallengeAccountState, pdt: PDTTracker) -> None:
        self._ensure_dir()
        with open(self._state_path, "w") as fh:
            json.dump(self._state_to_dict(state), fh, indent=2)
        with open(self._pdt_path, "w") as fh:
            json.dump({"day_trades_used_last_5d": pdt.day_trades_used_last_5d}, fh, indent=2)

    def reset(self) -> None:
        """Reset to clean starting state."""
        self._ensure_dir()
        fresh = ChallengeAccountState()
        fresh_pdt = PDTTracker()
        self.save(fresh, fresh_pdt)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_state(self) -> ChallengeAccountState:
        if not self._state_path.exists():
            return ChallengeAccountState()
        with open(self._state_path) as fh:
            d = json.load(fh)
        s = ChallengeAccountState()
        s.current_equity = float(d.get("current_equity", s.current_equity))
        s.peak_equity = float(d.get("peak_equity", s.peak_equity))
        s.milestone_stage = int(d.get("milestone_stage", s.milestone_stage))
        s.open_positions_count = int(d.get("open_positions_count", 0))
        s.realized_pnl = float(d.get("realized_pnl", 0.0))
        s.unrealized_pnl = float(d.get("unrealized_pnl", 0.0))
        s.sessions_run = int(d.get("sessions_run", 0))
        s.wins = int(d.get("wins", 0))
        s.losses = int(d.get("losses", 0))
        return s

    def _load_pdt(self) -> PDTTracker:
        if not self._pdt_path.exists():
            return PDTTracker()
        with open(self._pdt_path) as fh:
            d = json.load(fh)
        return PDTTracker(day_trades_used_last_5d=d.get("day_trades_used_last_5d", []))

    @staticmethod
    def _state_to_dict(s: ChallengeAccountState) -> dict:
        return {
            "current_equity": s.current_equity,
            "peak_equity": s.peak_equity,
            "milestone_stage": s.milestone_stage,
            "open_positions_count": s.open_positions_count,
            "realized_pnl": s.realized_pnl,
            "unrealized_pnl": s.unrealized_pnl,
            "sessions_run": s.sessions_run,
            "wins": s.wins,
            "losses": s.losses,
        }
