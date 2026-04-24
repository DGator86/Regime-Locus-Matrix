"""Data utilities: option chains, market datasets, bar enrichment, and microstructure.

Merged from former top-level packages: data, datasets (bar enrichment), microstructure.
"""

from rlm.data.liquidity_universe import (
    CORE_LIQUID_ETFS,
    EXPANDED_LIQUID_UNIVERSE,
    LIQUID_STOCK_EXTRAS,
    LIQUID_STOCK_UNIVERSE_10,
    LIQUID_UNIVERSE,
    MAGNIFICENT_SEVEN,
)
from rlm.data.massive import MassiveClient, load_massive_api_key
from rlm.data.massive_stocks import (
    aggregate_trade_flow_to_bars,
    bars_with_flow_from_massive,
    collect_paged_results,
    collect_stock_quotes,
    collect_stock_trades,
    massive_aggs_payload_to_bars_df,
    massive_quotes_payload_to_dataframe,
    massive_trades_payload_to_dataframe,
    trades_tick_rule_buy_sell,
)
from rlm.data.massive_option_chain import (
    collect_option_snapshot_pages,
    massive_option_chain_from_client,
    massive_option_snapshot_payload_to_dataframe,
    massive_option_snapshot_results_to_dataframe,
    massive_option_snapshot_to_normalized_chain,
)
from rlm.data.occ_symbol import ParsedOccSymbol, parse_occ_option_symbol

__all__ = [
    "CORE_LIQUID_ETFS",
    "EXPANDED_LIQUID_UNIVERSE",
    "LIQUID_STOCK_EXTRAS",
    "LIQUID_STOCK_UNIVERSE_10",
    "LIQUID_UNIVERSE",
    "MAGNIFICENT_SEVEN",
    "MassiveClient",
    "ParsedOccSymbol",
    "aggregate_trade_flow_to_bars",
    "bars_with_flow_from_massive",
    "collect_option_snapshot_pages",
    "collect_paged_results",
    "collect_stock_quotes",
    "collect_stock_trades",
    "load_massive_api_key",
    "massive_aggs_payload_to_bars_df",
    "massive_option_chain_from_client",
    "massive_quotes_payload_to_dataframe",
    "massive_trades_payload_to_dataframe",
    "massive_option_snapshot_payload_to_dataframe",
    "massive_option_snapshot_results_to_dataframe",
    "massive_option_snapshot_to_normalized_chain",
    "parse_occ_option_symbol",
    "trades_tick_rule_buy_sell",
]
