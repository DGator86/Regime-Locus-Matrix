"""rlm.challenge — $1K→$25K aggressive options dry-run challenge.

A self-contained simulation that runs on top of the persona pipeline.
Separate from IBKR equities and the standard options universe.

Quick start::

    # Reset (first time)
    rlm challenge --reset --symbol SPY

    # Run a session (fetches live data + persona interpretation)
    rlm challenge --run --symbol SPY

    # Check progress
    rlm challenge --status

Python API::

    from rlm.challenge import ChallengeConfig, ChallengeEngine, ChallengeTracker

    cfg = ChallengeConfig(seed_capital=1_000.0, target_capital=25_000.0)
    tracker = ChallengeTracker(data_root="data")
    engine = ChallengeEngine(cfg, tracker)

    summary = engine.run_session(
        directive="long",
        underlying_price=530.0,
        signal_alignment=0.81,
        confidence=0.74,
    )
    print(summary.message)
    print(f"Balance: ${summary.balance_after:,.2f}")
"""

from rlm.challenge.config import ChallengeConfig, ChallengeMilestone, MILESTONES
from rlm.challenge.engine import ChallengeEngine, SessionSummary
from rlm.challenge.state import ChallengePosition, ChallengeState, ChallengeTradeRecord
from rlm.challenge.tracker import ChallengeTracker

__all__ = [
    "ChallengeConfig",
    "ChallengeMilestone",
    "ChallengeEngine",
    "ChallengePosition",
    "ChallengeState",
    "ChallengeTradeRecord",
    "ChallengeTracker",
    "MILESTONES",
    "SessionSummary",
]
