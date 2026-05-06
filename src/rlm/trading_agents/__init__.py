"""TradingAgents integration for the RLM Hermes crew.

Requires the [trading_agents] optional dependency group:
    pip install -e ".[trading_agents]"
"""

from rlm.trading_agents.config import TradingAgentsConfig
from rlm.trading_agents.integration import TradingAgentsAdapter, TradingAgentsResult

__all__ = ["TradingAgentsConfig", "TradingAgentsAdapter", "TradingAgentsResult"]
