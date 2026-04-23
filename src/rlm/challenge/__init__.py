"""
Challenge module — $1,000 → $25,000 PDT-aware dry-run options growth engine.

Isolated from production IBKR flows.  All state stored under data/challenge/.
"""

from rlm.challenge.models import (
    ChallengeAccountState,
    ChallengeConfig,
    ChallengeDirective,
    ContractProfileRecommendation,
    PDTTracker,
    RiskPlan,
    SetupScoreResult,
    TradeModeDecision,
)
from rlm.challenge.pipeline import ChallengeDecisionPipeline
from rlm.challenge.state import ChallengeStateManager

__all__ = [
    "ChallengeConfig",
    "ChallengeAccountState",
    "PDTTracker",
    "SetupScoreResult",
    "TradeModeDecision",
    "ContractProfileRecommendation",
    "RiskPlan",
    "ChallengeDirective",
    "ChallengeDecisionPipeline",
    "ChallengeStateManager",
]
