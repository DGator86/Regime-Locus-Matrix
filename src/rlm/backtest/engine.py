from __future__ import annotations

import pandas as pd

from rlm.backtest.portfolio import Portfolio
from rlm.data.option_chain import normalize_option_chain, select_nearest_expiry_slice
from rlm.roee.chain_match import match_legs_to_chain
from rlm.roee.exits import should_exit_for_profit, should_exit_for_regime_flip, should_exit_for_zone_breach
from rlm.roee.policy import select_trade
from rlm.scoring.state_matrix import classify_state_matrix


class BacktestEngine:
    def __init__(
        self,
        *,
        initial_capital: float = 100_000.0,
        contract_multiplier: int = 100,
        strike_increment: float = 1.0,
        underlying_symbol: str = "SPY",
        quantity_per_trade: int = 1,
    ) -> None:
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            contract_multiplier=contract_multiplier,
        )
        self.strike_increment = strike_increment
        self.underlying_symbol = underlying_symbol
        self.quantity_per_trade = quantity_per_trade

    def run(
        self,
        feature_df: pd.DataFrame,
        option_chain_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        chain = normalize_option_chain(option_chain_df)
        features = classify_state_matrix(feature_df.copy())

        for ts, row in features.iterrows():
            row_chain = chain[
                (chain["timestamp"] == pd.Timestamp(ts))
                & (chain["underlying"] == self.underlying_symbol)
            ].copy()

            self._process_exits(ts, row)

            if row_chain.empty:
                self.portfolio.mark_equity(ts)
                continue

            decision = select_trade(
                current_price=float(row["close"]),
                sigma=float(row["sigma"]),
                s_d=float(row["S_D"]),
                s_v=float(row["S_V"]),
                s_l=float(row["S_L"]),
                s_g=float(row["S_G"]),
                direction_regime=str(row["direction_regime"]),
                volatility_regime=str(row["volatility_regime"]),
                liquidity_regime=str(row["liquidity_regime"]),
                dealer_flow_regime=str(row["dealer_flow_regime"]),
                regime_key=str(row["regime_key"]),
                bid_ask_spread_pct=float(row["bid_ask_spread"] / row["close"])
                if "bid_ask_spread" in row and pd.notna(row["bid_ask_spread"])
                else None,
                has_major_event=bool(row["has_major_event"])
                if "has_major_event" in row and pd.notna(row["has_major_event"])
                else False,
                strike_increment=self.strike_increment,
            )

            if decision.action == "enter":
                dte_min = int(decision.candidate.target_dte_min) if decision.candidate else 20
                dte_max = int(decision.candidate.target_dte_max) if decision.candidate else 45
                chain_slice = select_nearest_expiry_slice(row_chain, dte_min=dte_min, dte_max=dte_max)

                if not chain_slice.empty:
                    matched_decision = match_legs_to_chain(decision=decision, chain_slice=chain_slice)
                    if matched_decision.action == "enter":
                        self.portfolio.open_from_decision(
                            timestamp=pd.Timestamp(ts),
                            underlying_symbol=self.underlying_symbol,
                            underlying_price=float(row["close"]),
                            decision=matched_decision,
                            quantity=self.quantity_per_trade,
                        )

            self.portfolio.mark_equity(ts)

        equity_frame = self.portfolio.equity_frame()
        trades_frame = self.portfolio.closed_trades_frame()

        from rlm.backtest.metrics import summarize_backtest

        summary = summarize_backtest(equity_frame, trades_frame)
        return equity_frame, trades_frame, summary

    def _process_exits(self, ts: pd.Timestamp, row: pd.Series) -> None:
        to_close: list[tuple[str, str]] = []

        for position_id, pos in self.portfolio.open_positions.items():
            pnl_pct = pos.pnl_pct()

            if should_exit_for_profit(pnl_pct=pnl_pct, target_profit_pct=pos.target_profit_pct):
                to_close.append((position_id, "profit_target"))
                continue

            if should_exit_for_zone_breach(
                realized_price=float(row["close"]),
                lower_1s=float(row["lower_1s"]),
                upper_1s=float(row["upper_1s"]),
            ):
                to_close.append((position_id, "zone_breach"))
                continue

            if should_exit_for_regime_flip(
                entry_regime_key=pos.regime_key,
                current_regime_key=str(row["regime_key"]),
            ):
                to_close.append((position_id, "regime_flip"))
                continue

        for position_id, reason in to_close:
            self.portfolio.close_position(
                position_id=position_id,
                timestamp_exit=pd.Timestamp(ts),
                underlying_price=float(row["close"]),
                exit_reason=reason,
            )
