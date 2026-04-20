"""TradeService — application layer for live/paper trade plan generation and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline
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


@dataclass
class TradeResult:
    decision: dict | None = None
    execution_log: list[str] = field(default_factory=list)


class TradeService:
    """Generate a trade decision from the latest market data and optionally execute it.

    In ``plan`` mode the decision is printed only — no orders are placed.
    In ``paper`` / ``live`` mode the decision is forwarded to the IBKR execution layer.
    """

    def run(self, req: TradeRequest) -> TradeResult:
        log.info("TradeService.run symbol=%s mode=%s", req.symbol, req.mode)

        bars_df = req.bars_df
        chain_df = req.option_chain_df

        if bars_df is None:
            bars_df = self._load_latest_bars(req.symbol)
        if chain_df is None:
            chain_df = self._load_latest_chain(req.symbol)

        cfg = FullRLMConfig(
            symbol=req.symbol,
            use_kronos=req.use_kronos,
            attach_vix=req.attach_vix,
            initial_capital=req.capital,
        )

        pipeline = FullRLMPipeline(cfg)
        result = pipeline.run(bars_df, chain_df)

        if result.policy_df.empty:
            return TradeResult(execution_log=["No policy rows — cannot generate decision."])

        last_row = result.policy_df.iloc[-1].to_dict()
        decision = {
            "roee_action": last_row.get("roee_action"),
            "roee_strategy": last_row.get("roee_strategy"),
            "roee_size_fraction": last_row.get("roee_size_fraction"),
            "vault_triggered": last_row.get("vault_triggered"),
        }

        log.info("Decision: %s", decision)

        if req.mode in ("paper", "live"):
            execution_log = self._execute(req, decision)
        else:
            execution_log = ["Plan mode — no orders placed."]

        return TradeResult(decision=decision, execution_log=execution_log)

    def _load_latest_bars(self, symbol: str) -> pd.DataFrame:
        root = Path(__file__).resolve().parents[5]
        bars_path = root / f"data/raw/bars_{symbol}.csv"
        if not bars_path.is_file():
            raise FileNotFoundError(
                f"No bars file found for {symbol}. Run: rlm ingest --symbol {symbol}"
            )
        df = pd.read_csv(bars_path, parse_dates=["timestamp"])
        return df.sort_values("timestamp").set_index("timestamp")

    def _load_latest_chain(self, symbol: str) -> pd.DataFrame | None:
        root = Path(__file__).resolve().parents[5]
        chain_path = root / f"data/raw/option_chain_{symbol}.csv"
        if not chain_path.is_file():
            return None
        return pd.read_csv(chain_path, parse_dates=["timestamp", "expiry"])

    def _execute(self, req: TradeRequest, decision: dict) -> list[str]:
        action = decision.get("roee_action", "hold")
        if action == "hold":
            return ["Action=hold — no orders submitted."]

        try:
            from rlm.execution.ibkr_combo import place_roee_combo

            log.info("Placing %s order via IBKR (%s mode)", action, req.mode)
            order_id = place_roee_combo(
                symbol=req.symbol,
                decision=decision,
                paper=(req.mode == "paper"),
            )
            return [f"Order placed: id={order_id} action={action}"]
        except Exception as exc:
            log.error("IBKR execution failed: %s", exc)
            return [f"Execution failed: {exc}"]
