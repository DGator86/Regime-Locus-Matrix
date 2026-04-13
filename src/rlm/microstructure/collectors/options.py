"""
Real-time option chain collector with full Greek calculation.

Fetches the entire option chain for a given underlying from Interactive Brokers,
computes all 13 Black-Scholes Greeks (including second- and third-order) from
the mid-price, and persists 5-second-interval snapshots as date-partitioned Parquet.

Chain filtering
---------------
The full SPX/SPY chain has thousands of contracts.  To stay within IBKR's
market-data line limits we:
  1. Restrict to strikes within ``strike_band_pct`` of ATM (default ±15%)
  2. Restrict to expirations within ``max_dte`` days (default 90)
  3. Apply a minimum open-interest filter (default 10 contracts)

Usage::

    # CLI — run once (snapshot)
    python -m rlm.microstructure.collectors.options --symbol SPY

    # CLI — continuous 5-second snapshots
    python -m rlm.microstructure.collectors.options --symbol SPY --continuous --interval 5

    # Programmatic
    from rlm.microstructure.collectors.options import OptionsCollector
    collector = OptionsCollector("SPY")
    df = collector.fetch_snapshot()          # one-shot
    await collector.stream(interval=5)       # continuous
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlm.data.ibkr_stocks import load_ibkr_socket_config
from rlm.microstructure.calculators.greeks import compute_greeks_dataframe, solve_iv

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Contract, ContractDetails, Option, Stock, util
    _HAS_IB_INSYNC = True
except ImportError:
    _HAS_IB_INSYNC = False


_SNAPSHOT_COLS = [
    "timestamp",
    "underlying_symbol",
    "underlying_price",
    "contract_symbol",
    "expiration",
    "dte",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "volume",
    "open_interest",
    "implied_vol",
    # First-order
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    # Second-order
    "vanna",
    "charm",
    "vomma",
    "veta",
    # Third-order
    "speed",
    "zomma",
    "color",
    "ultima",
]


def _snap_path(symbol: str, date: Any, data_root: str) -> Path:
    return Path(data_root) / f"options/{symbol}/greeks_snapshots/{symbol}_{date}.parquet"


def _append_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp", "contract_symbol"]).sort_values(
            ["timestamp", "contract_symbol"]
        )
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class OptionsCollector:
    """
    Fetches option chain data from IBKR and computes the full Greek surface.

    Parameters
    ----------
    symbol              : Underlying ticker (e.g. "SPY")
    data_root           : Root of microstructure lake
    strike_band_pct     : Only include strikes within this % of ATM (e.g. 0.15 = ±15%)
    max_dte             : Maximum days to expiry included
    min_open_interest   : Filter contracts below this OI threshold
    risk_free           : Risk-free rate for Greek calculations
    client_id           : IBKR client ID offset (added to config value for market data)
    """

    def __init__(
        self,
        symbol: str,
        *,
        data_root: str = "data/microstructure",
        strike_band_pct: float = 0.15,
        max_dte: int = 90,
        min_open_interest: int = 10,
        risk_free: float = 0.052,
        client_id: int | None = None,
    ) -> None:
        if not _HAS_IB_INSYNC:
            raise ImportError(
                "ib_insync is required for OptionsCollector.\n"
                "Install with: pip install 'regime-locus-matrix[ib-insync]'"
            )
        self.symbol = symbol.upper()
        self.data_root = data_root
        self.strike_band_pct = strike_band_pct
        self.max_dte = max_dte
        self.min_open_interest = min_open_interest
        self.risk_free = risk_free
        self._host, self._port, self._cid = load_ibkr_socket_config()
        if client_id is not None:
            self._cid = client_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self, id_offset: int = 0) -> IB:
        ib = IB()
        ib.connect(self._host, self._port, clientId=self._cid + id_offset, timeout=60)
        return ib

    def _get_underlying_price(self, ib: IB) -> float:
        contract = Stock(self.symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract, "", False, False)
        ib.sleep(2)
        price = ticker.last or ticker.close or float("nan")
        ib.cancelMktData(contract)
        return float(price)

    def _get_option_contracts(self, ib: IB, spot: float) -> list[Any]:
        """Fetch contract details for ATM-band options within max_dte."""
        chain = ib.reqSecDefOptParams(self.symbol, "", "STK", 0)
        if not chain:
            return []

        exch_chain = chain[0]
        today = datetime.utcnow().date()
        max_date = today + timedelta(days=self.max_dte)
        lo_strike = spot * (1 - self.strike_band_pct)
        hi_strike = spot * (1 + self.strike_band_pct)

        contracts = []
        for expiry_str in exch_chain.expirations:
            try:
                exp_date = datetime.strptime(expiry_str, "%Y%m%d").date()
            except ValueError:
                continue
            if exp_date <= today or exp_date > max_date:
                continue
            for strike in exch_chain.strikes:
                if not (lo_strike <= strike <= hi_strike):
                    continue
                for right in ("C", "P"):
                    contracts.append(
                        Option(self.symbol, expiry_str, strike, right, "SMART", "USD")
                    )
        return contracts

    def _fetch_market_data(self, ib: IB, contracts: list[Any], spot: float) -> pd.DataFrame:
        """Request market data for all contracts and build raw snapshot DataFrame."""
        # Qualify in batches (IBKR limit: 200 simultaneous market data lines)
        BATCH = 100
        records: list[dict[str, Any]] = []
        now = pd.Timestamp.utcnow().tz_localize(None)
        today = now.date()

        for i in range(0, len(contracts), BATCH):
            batch = contracts[i : i + BATCH]
            ib.qualifyContracts(*batch)
            tickers = {c: ib.reqMktData(c, "101", False, False) for c in batch}
            ib.sleep(3)  # allow market data to populate

            for contract, ticker in tickers.items():
                bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else float("nan")
                ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else float("nan")
                mid = (bid + ask) / 2.0 if not (np.isnan(bid) or np.isnan(ask)) else float("nan")
                oi = int(ticker.callOpenInterest or ticker.putOpenInterest or 0)
                volume = int(ticker.volume or 0)

                if np.isnan(mid) or oi < self.min_open_interest:
                    ib.cancelMktData(contract)
                    continue

                try:
                    exp_date = datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d").date()
                except (ValueError, AttributeError):
                    ib.cancelMktData(contract)
                    continue

                dte = (exp_date - today).days
                is_call = contract.right.upper() == "C"
                iv = solve_iv(
                    market_price=mid,
                    spot=spot,
                    strike=float(contract.strike),
                    time_years=dte / 365.0,
                    is_call=is_call,
                    risk_free=self.risk_free,
                )

                records.append({
                    "timestamp": now,
                    "underlying_symbol": self.symbol,
                    "underlying_price": spot,
                    "contract_symbol": contract.localSymbol or f"{self.symbol}_{contract.lastTradeDateOrContractMonth}_{contract.strike}{contract.right}",
                    "expiration": exp_date,
                    "dte": dte,
                    "strike": float(contract.strike),
                    "option_type": "call" if is_call else "put",
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "volume": volume,
                    "open_interest": oi,
                    "implied_vol": iv,
                })
                ib.cancelMktData(contract)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_snapshot(self) -> pd.DataFrame:
        """
        Fetch one full option chain snapshot and compute all Greeks.

        Returns
        -------
        DataFrame with all columns defined in ``_SNAPSHOT_COLS``.
        Saves to date-partitioned Parquet.
        """
        ib = self._connect(id_offset=20)
        try:
            spot = self._get_underlying_price(ib)
            if np.isnan(spot):
                logger.error("Could not determine spot price for %s", self.symbol)
                return pd.DataFrame(columns=_SNAPSHOT_COLS)

            contracts = self._get_option_contracts(ib, spot)
            if not contracts:
                logger.warning("No option contracts found for %s", self.symbol)
                return pd.DataFrame(columns=_SNAPSHOT_COLS)

            logger.info("Requesting market data for %d contracts (%s)", len(contracts), self.symbol)
            raw = self._fetch_market_data(ib, contracts, spot)
        finally:
            ib.disconnect()

        if raw.empty:
            return pd.DataFrame(columns=_SNAPSHOT_COLS)

        # Compute full Greek suite
        enriched = compute_greeks_dataframe(raw, risk_free=self.risk_free)

        # Ensure all schema columns present
        for col in _SNAPSHOT_COLS:
            if col not in enriched.columns:
                enriched[col] = float("nan")

        enriched = enriched[_SNAPSHOT_COLS].copy()
        self._save(enriched)
        logger.info("Snapshot saved: %d contracts for %s @ %.2f", len(enriched), self.symbol, spot)
        return enriched

    async def stream(self, *, interval: float = 5.0, max_runtime_hours: float = 7.0) -> None:
        """
        Continuously fetch option snapshots every *interval* seconds.

        Parameters
        ----------
        interval            : Seconds between snapshots (default 5)
        max_runtime_hours   : Hard stop after this many hours
        """
        logger.info("Starting option chain stream for %s (interval=%ss)", self.symbol, interval)
        deadline = time.monotonic() + max_runtime_hours * 3600

        while time.monotonic() < deadline:
            try:
                df = self.fetch_snapshot()
                logger.debug("Option snapshot: %d rows", len(df))
            except Exception as exc:
                logger.error("Option snapshot failed: %s", exc)
            await asyncio.sleep(interval)

        logger.info("Option stream stopped for %s", self.symbol)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
        for date, group in df.groupby("_date"):
            path = _snap_path(self.symbol, date, self.data_root)
            _append_parquet(group.drop(columns="_date"), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect IBKR option chain snapshots with full Greeks into the microstructure lake."
    )
    parser.add_argument("--symbol", required=True, help="Underlying ticker (e.g. SPY)")
    parser.add_argument(
        "--continuous", action="store_true", help="Run continuously (default: single snapshot)"
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between snapshots")
    parser.add_argument(
        "--strike-band", type=float, default=0.15, help="ATM band as decimal (default 0.15 = ±15%%)"
    )
    parser.add_argument("--max-dte", type=int, default=90, help="Maximum DTE to include")
    parser.add_argument("--min-oi", type=int, default=10, help="Minimum open interest filter")
    parser.add_argument("--data-root", default="data/microstructure")
    args = parser.parse_args()

    collector = OptionsCollector(
        args.symbol,
        data_root=args.data_root,
        strike_band_pct=args.strike_band,
        max_dte=args.max_dte,
        min_open_interest=args.min_oi,
    )

    if args.continuous:
        asyncio.run(collector.stream(interval=args.interval))
    else:
        df = collector.fetch_snapshot()
        print(f"Snapshot: {len(df)} contracts for {args.symbol}")
        if not df.empty:
            print(df[["strike", "option_type", "dte", "mid", "implied_vol", "delta", "gamma"]].head(10).to_string())


if __name__ == "__main__":
    _main()
