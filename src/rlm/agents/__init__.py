"""
Star Trek AI crew for Regime Locus Matrix.

- Scotty  : system health guardian — keeps the engines running
- Spock   : logical market analyst — gives probability-weighted advice
- Kirk    : strategy commander — synthesises crew reports, drives decisions
- Crew    : orchestrator that runs all three on schedule
"""

from rlm.agents.base import LLMClient, LLMConfig, Message
from rlm.agents.crew import StarfleetCrew
from rlm.agents.kirk import KirkAgent
from rlm.agents.scotty import ScottyAgent
from rlm.agents.spock import SpockAgent

__all__ = [
    "LLMClient",
    "LLMConfig",
    "Message",
    "ScottyAgent",
    "SpockAgent",
    "KirkAgent",
    "StarfleetCrew",
]
