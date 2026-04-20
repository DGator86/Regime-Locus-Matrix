"""TradeService — application layer for live/paper trade plan generation and execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline
from rlm.data.readers import load_bars, load_option_chain
from rlm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TradeRequest:
    """Input bundle for a trade plan / execution run."""

    symbol: str
    mode: str = "plan"  # "plan" | "paper" | "live"
    use_kronos: bool = True
    attach_vix: bool = True
    capital: float = 100_000.0
    bars_df: pd.DataFrame | None = None
    option_chain_df: pd.DataFrame | None = None
    data_root: str | None = None  # maps to --data-root / RLM_DATA_ROOT


@dataclass
class TradeResult:
    decision: dict | None = None
    execution_log: list[str] = field(default_factory=list)
    duration_s: float = 0.0


class TradeService:
    """Generate a trade decision from the latest market data and optionally execute it.

    In ``plan`` mode the decision is printed only — no orders are placed.
    In ``paper`` / ``live`` mode the decision is forwarded to the IBKR execution layer.

    Data loading uses ``rlm.data.readers`` so no ``__file__``-relative paths appear here.
    """

    def run(self, req: TradeRequest) -> TradeResult:
        t0 = time.monotonic()
        log.info("trade start  symbol=%s mode=%s kronos=%s", req.symbol, req.mode, req.use_kronos)

        bars_df = req.bars_df
        chain_df = req.option_chain_df

        if bars_df is None:
            bars_df = load_bars(req.symbol, data_root=req.data_root)
        if chain_df is None:
            chain_df = load_option_chain(req.symbol, data_root=req.data_root)

        cfg = FullRLMConfig(
            symbol=req.symbol,
            use_kronos=req.use_kronos,
            attach_vix=req.attach_vix,
            initial_capital=req.capital,
        )

        result = FullRLMPipeline(cfg).run(bars_df, chain_df)

        if result.policy_df.empty:
            log.warning("trade no policy rows  symbol=%s", req.symbol)
            return TradeResult(
                execution_log=["No policy rows — cannot generate decision."],
                duration_s=time.monotonic() - t0,
            )

        last_row = result.policy_df.iloc[-1].to_dict()
        decision = {
            "roee_action": last_row.get("roee_action"),
            "roee_strategy": last_row.get("roee_strategy"),
            "roee_size_fraction": last_row.get("roee_size_fraction"),
            "vault_triggered": last_row.get("vault_triggered"),
        }

        log.info(
            "trade decision  symbol=%s action=%s strategy=%s size=%s",
            req.symbol, decision["roee_action"], decision["roee_strategy"], decision["roee_size_fraction"],
        )

        if req.mode in ("paper", "live"):
            execution_log = self._execute(req, decision)
        else:
            execution_log = ["Plan mode — no orders placed."]

        return TradeResult(
            decision=decision,
            execution_log=execution_log,
            duration_s=time.monotonic() - t0,
        )

    def _execute(self, req: TradeRequest, decision: dict) -> list[str]:
        action = decision.get("roee_action", "hold")
        if action == "hold":
            return ["Action=hold — no orders submitted."]

        try:
            from rlm.execution.ibkr_combo import place_roee_combo

            log.info("ibkr order  symbol=%s action=%s mode=%s", req.symbol, action, req.mode)
            order_id = place_roee_combo(
                symbol=req.symbol,
                decision=decision,
                paper=(req.mode == "paper"),
            )
            return [f"Order placed: id={order_id} action={action}"]
        except Exception as exc:
            log.error("ibkr order failed  symbol=%s error=%s", req.symbol, exc)
            return [f"Execution failed: {exc}"]
