"""ChallengeState — persisted account and position state for the dry-run challenge."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from rlm.challenge.config import MILESTONES, ChallengeConfig


@dataclass
class ChallengePosition:
    """A single open dry-run option position."""

    position_id: str
    symbol: str
    option_type: Literal["call", "put"]
    direction: Literal["long", "short"]
    underlying_entry: float
    strike: float
    dte_at_entry: int
    entry_date: str
    premium_per_share: float
    """Option price per share at entry (contracts × 100 = total notional per contract)."""
    qty: int
    """Number of option contracts held."""
    total_cost: float
    """Actual cash spent: premium_per_share × qty × 100."""
    delta_at_entry: float
    iv_at_entry: float

    # Updated each session
    dte_remaining: int = 0
    current_premium: float = 0.0
    current_value: float = 0.0
    unrealised_pnl: float = 0.0
    status: Literal["open", "closed"] = "open"

    @classmethod
    def new(
        cls,
        *,
        symbol: str,
        option_type: Literal["call", "put"],
        direction: Literal["long", "short"],
        underlying_entry: float,
        strike: float,
        dte: int,
        entry_date: str,
        premium_per_share: float,
        qty: int,
        delta: float,
        iv: float,
    ) -> "ChallengePosition":
        total = premium_per_share * qty * 100
        return cls(
            position_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            option_type=option_type,
            direction=direction,
            underlying_entry=underlying_entry,
            strike=strike,
            dte_at_entry=dte,
            entry_date=entry_date,
            premium_per_share=premium_per_share,
            qty=qty,
            total_cost=total,
            delta_at_entry=delta,
            iv_at_entry=iv,
            dte_remaining=dte,
            current_premium=premium_per_share,
            current_value=total,
            unrealised_pnl=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChallengePosition":
        return cls(**d)


@dataclass
class ChallengeTradeRecord:
    """Closed trade record appended to the trade log."""

    trade_id: str
    symbol: str
    option_type: str
    direction: str
    strike: float
    dte_at_entry: int
    entry_date: str
    exit_date: str
    premium_paid: float
    """Total cash spent at entry."""
    proceeds: float
    """Total cash received at exit."""
    pnl: float
    pnl_pct: float
    exit_reason: Literal["target", "stop", "expiry", "manual"]
    balance_before: float
    balance_after: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ChallengeState:
    """Full mutable state of the challenge, persisted between sessions."""

    balance: float
    seed: float
    target: float
    open_positions: list[ChallengePosition] = field(default_factory=list)
    trade_history: list[ChallengeTradeRecord] = field(default_factory=list)
    session_count: int = 0
    created_at: str = ""
    last_updated: str = ""

    # ---- Derived properties -------------------------------------------------

    @property
    def progress_pct(self) -> float:
        """Fraction of the journey from seed to target completed (0–1)."""
        span = self.target - self.seed
        if span <= 0:
            return 1.0
        return min(1.0, max(0.0, (self.balance - self.seed) / span))

    @property
    def current_milestone_idx(self) -> int:
        """Index of the next un-cleared milestone (0-based)."""
        for i, m in enumerate(MILESTONES):
            if self.balance < m.target:
                return i
        return len(MILESTONES) - 1

    @property
    def current_milestone(self) -> object:
        return MILESTONES[min(self.current_milestone_idx, len(MILESTONES) - 1)]

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trade_history if t.pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trade_history if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        total = len(self.trade_history)
        return self.wins / total if total else 0.0

    @property
    def total_return_pct(self) -> float:
        return (self.balance - self.seed) / self.seed * 100.0

    # ---- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "balance": self.balance,
            "seed": self.seed,
            "target": self.target,
            "session_count": self.session_count,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "open_positions": [p.to_dict() for p in self.open_positions],
            "trade_history": [t.to_dict() for t in self.trade_history],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChallengeState":
        positions = [ChallengePosition.from_dict(p) for p in d.get("open_positions", [])]
        trades = [ChallengeTradeRecord(**t) for t in d.get("trade_history", [])]
        return cls(
            balance=float(d["balance"]),
            seed=float(d["seed"]),
            target=float(d["target"]),
            open_positions=positions,
            trade_history=trades,
            session_count=int(d.get("session_count", 0)),
            created_at=str(d.get("created_at", "")),
            last_updated=str(d.get("last_updated", "")),
        )

    @classmethod
    def fresh(cls, cfg: ChallengeConfig, now: str) -> "ChallengeState":
        return cls(
            balance=cfg.seed_capital,
            seed=cfg.seed_capital,
            target=cfg.target_capital,
            created_at=now,
            last_updated=now,
        )
