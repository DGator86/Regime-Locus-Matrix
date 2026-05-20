"""rlm.challenge — $1K→$25K aggressive options dry-run challenge.

A self-contained simulation that runs on top of the persona pipeline.
Separate from IBKR equities and the standard options universe.

Quick start::

    # Reset (first time) — underlyings are SPY and QQQ only
    rlm challenge --reset --symbol SPY

    # Run a session (fetches live data + persona interpretation)
    rlm challenge --run --symbol SPY

    # Check progress
    rlm challenge --status

``rlm challenge --run`` may still open positions when persona vetoes directional
trades — if ``RLM_CHALLENGE_REGIME_FALLBACK`` is enabled (default), the engine
reads ``direction_regime`` / ``regime_key`` / latent HMM labels on the ROEE row
(so the dry-run book reflects regime bias). Set ``RLM_CHALLENGE_REGIME_FALLBACK=0``
to keep strict persona-only behaviour.

Dashboard / local workstation::

    # Refresh processed + challenge state from VPS (needs SSH/scp keys); from repo root:
    #   powershell ./scripts/sync_dashboard_data_from_vps.ps1

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

from rlm.challenge.challenge_strategy_map import STRATEGY_MAP_CHALLENGE, get_challenge_strategy
from rlm.challenge.config import MILESTONES, ChallengeConfig, ChallengeMilestone, apply_challenge_profile_env
from rlm.challenge.daytrade_filters import get_iv_rank, is_great_daytrade_setup
from rlm.challenge.engine import ChallengeEngine, SessionSummary
from rlm.challenge.models import ChallengePipelineConfig
from rlm.challenge.pipeline import ChallengeDecisionPipeline
from rlm.challenge.state import ChallengePosition, ChallengeState, ChallengeTradeRecord
from rlm.challenge.symbols import (
    CHALLENGE_ALLOWED_UNDERLYINGS,
    normalize_challenge_underlying,
    resolve_challenge_underlyings_from_environ,
)
from rlm.challenge.tracker import ChallengeTracker

__all__ = [
    "CHALLENGE_ALLOWED_UNDERLYINGS",
    "ChallengeConfig",
    "ChallengeDecisionPipeline",
    "ChallengeMilestone",
    "ChallengePipelineConfig",
    "ChallengeEngine",
    "ChallengePosition",
    "ChallengeState",
    "ChallengeTradeRecord",
    "ChallengeTracker",
    "MILESTONES",
    "STRATEGY_MAP_CHALLENGE",
    "SessionSummary",
    "apply_challenge_profile_env",
    "get_challenge_strategy",
    "get_iv_rank",
    "is_great_daytrade_setup",
    "normalize_challenge_underlying",
    "resolve_challenge_underlyings_from_environ",
]
