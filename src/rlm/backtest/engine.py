from __future__ import annotations

import pandas as pd

from rlm.backtest.fills import FillConfig
from rlm.backtest.lifecycle import (
    ExpiryLiquidationPolicy,
    LifecycleConfig,
    is_at_or_past_expiry,
    should_close_for_max_holding,
    should_force_close_before_expiry,
)
from rlm.backtest.portfolio import Portfolio
from rlm.data.option_chain import normalize_option_chain, select_nearest_expiry_slice
from rlm.roee.chain_match import match_legs_to_chain
from rlm.roee.decision import select_trade_for_row
from rlm.roee.exits import should_exit_for_profit, should_exit_for_regime_flip, should_exit_for_zone_breach
from rlm.roee.pipeline import ROEEConfig
from rlm.scoring.state_matrix import classify_state_matrix

# Alias for tests and external monkeypatching (backtests call this once per bar).
decide_trade_for_bar = select_trade_for_row


class BacktestEngine:
    def __init__(
        self,
        *,
        initial_capital: float = 100_000.0,
        contract_multiplier: int = 100,
        strike_increment: float = 1.0,
        underlying_symbol: str = "SPY",
        quantity_per_trade: int = 1,
        fill_config: FillConfig | None = None,
        lifecycle_config: LifecycleConfig | None = None,
        roee_config: ROEEConfig | None = None,
    ) -> None:
        self.lifecycle_config = lifecycle_config or LifecycleConfig()
        self.roee_config = roee_config
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            contract_multiplier=contract_multiplier,
            lifecycle_config=self.lifecycle_config,
        )
        self.strike_increment = strike_increment
        self.underlying_symbol = underlying_symbol
        self.quantity_per_trade = quantity_per_trade
        self.fill_config = fill_config or FillConfig(contract_multiplier=contract_multiplier)

    def run(
        self,
        feature_df: pd.DataFrame,
        option_chain_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        chain = normalize_option_chain(option_chain_df)
        features = classify_state_matrix(feature_df.copy())

        ch = chain.loc[chain["underlying"] == self.underlying_symbol].copy()
        chain_by_ts: dict[pd.Timestamp, pd.DataFrame] = {}
        for tkey, grp in ch.groupby("timestamp", sort=False):
            chain_by_ts[pd.Timestamp(tkey)] = grp.copy()

        for bar_index, (ts, row) in enumerate(features.iterrows()):
            traded_this_bar = False
            row_chain = chain_by_ts.get(pd.Timestamp(ts), pd.DataFrame()).copy()

            if not row_chain.empty:
                self.portfolio.revalue_open_positions(
                    chain_snapshot=row_chain,
                    fill_config=self.fill_config,
                )

            self._process_exits(ts, row, bar_index)

            if row_chain.empty:
                self.portfolio.mark_equity(ts)
                continue

            rc = self.roee_config
            decision = decide_trade_for_bar(
                row,
                strike_increment=self.strike_increment,
                hmm_confidence_threshold=rc.hmm_confidence_threshold if rc is not None else None,
                hmm_sizing_multiplier=rc.sizing_multiplier if rc is not None else 1.0,
                hmm_transition_penalty=rc.transition_penalty if rc is not None else 0.5,
                use_dynamic_sizing=rc.use_dynamic_sizing if rc is not None else False,
                vol_target=rc.vol_target if rc is not None else 0.15,
                max_kelly_fraction=rc.max_kelly_fraction if rc is not None else 0.25,
                max_capital_fraction=rc.max_capital_fraction if rc is not None else 0.5,
                vault_uncertainty_threshold=rc.vault_uncertainty_threshold if rc is not None else 0.03,
                vault_size_multiplier=rc.vault_size_multiplier if rc is not None else 0.5,
            )

            if decision.action == "enter" and not (self.lifecycle_config.one_trade_per_bar and traded_this_bar):
                dte_min = int(decision.candidate.target_dte_min) if decision.candidate else 20
                dte_max = int(decision.candidate.target_dte_max) if decision.candidate else 45
                chain_slice = select_nearest_expiry_slice(row_chain, dte_min=dte_min, dte_max=dte_max)

                if not chain_slice.empty:
                    matched_decision = match_legs_to_chain(decision=decision, chain_slice=chain_slice)
                    if matched_decision.action == "enter":
                        opened_id = self.portfolio.open_from_decision(
                            timestamp=pd.Timestamp(ts),
                            underlying_symbol=self.underlying_symbol,
                            underlying_price=float(row["close"]),
                            decision=matched_decision,
                            quantity=self.quantity_per_trade,
                            fill_config=self.fill_config,
                            bar_index=bar_index,
                        )
                        traded_this_bar = opened_id is not None

            self.portfolio.mark_equity(ts)

        equity_frame = self.portfolio.equity_frame()
        trades_frame = self.portfolio.closed_trades_frame()

        from rlm.backtest.metrics import summarize_backtest
        summary = summarize_backtest(equity_frame, trades_frame)
        return equity_frame, trades_frame, summary

    def _process_exits(self, ts: pd.Timestamp, row: pd.Series, bar_index: int) -> None:
        to_close: list[tuple[str, str]] = []
        to_settle: list[str] = []

        policy = self.lifecycle_config.expiry_liquidation_policy

        for position_id, pos in self.portfolio.open_positions.items():
            if should_close_for_max_holding(
                entry_bar_index=pos.entry_bar_index,
                current_bar_index=bar_index,
                config=self.lifecycle_config,
            ):
                to_close.append((position_id, "max_holding"))
                continue

            at_expiry = is_at_or_past_expiry(
                timestamp=pd.Timestamp(ts),
                expiry=pos.expiry,
            )

            if policy == ExpiryLiquidationPolicy.SETTLE_AT_EXPIRY:
                # Under SETTLE_AT_EXPIRY: let the position reach expiry and
                # settle it using intrinsic value.  Do not force-close early.
                if at_expiry:
                    to_settle.append(position_id)
                    continue
            else:
                # Default LIQUIDATE_BEFORE_EXPIRY behaviour.
                if should_force_close_before_expiry(
                    timestamp=pd.Timestamp(ts),
                    expiry=pos.expiry,
                    config=self.lifecycle_config,
                ):
                    to_close.append((position_id, "forced_pre_expiry"))
                    continue

                if self.lifecycle_config.close_at_expiry_if_open and at_expiry:
                    to_close.append((position_id, "expiry_close"))
                    continue

            pricing_ok = bool(pos.metadata.get("reprice_ok", True))

            pnl_pct = pos.pnl_pct()

            if pricing_ok and should_exit_for_profit(pnl_pct=pnl_pct, target_profit_pct=pos.target_profit_pct):
                to_close.append((position_id, "profit_target"))
                continue

            if pricing_ok and should_exit_for_zone_breach(
                realized_price=float(row["close"]),
                lower_1s=float(row["lower_1s"]),
                upper_1s=float(row["upper_1s"]),
            ):
                to_close.append((position_id, "zone_breach"))
                continue

            if pricing_ok and should_exit_for_regime_flip(
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

        for position_id in to_settle:
            self.portfolio.expiry_settle_position(
                position_id=position_id,
                timestamp_exit=pd.Timestamp(ts),
                underlying_price=float(row["close"]),
            )
