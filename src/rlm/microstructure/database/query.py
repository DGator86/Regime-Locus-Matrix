"""
DuckDB query interface for the RLM microstructure data lake.

The ``MicrostructureDB`` class provides a unified, lazily-connected interface to
all microstructure Parquet files.  DuckDB reads the files directly (no import
step required) using glob patterns, which means the caller never loads an entire
partition into memory.

Lake layout expected::

    data/microstructure/
    ├── underlying/{symbol}/1s/{symbol}_{date}.parquet
    ├── options/{symbol}/
    │   ├── greeks_snapshots/{symbol}_{date}.parquet
    │   └── derived/
    │       ├── gex_surface/{symbol}_{date}.parquet
    │       └── iv_surface/{symbol}_{date}.parquet
    └── metadata/

Usage::

    from rlm.microstructure.database.query import MicrostructureDB

    db = MicrostructureDB()
    bars = db.load_underlying_bars("SPY", "2025-06-01", "2025-06-10")
    gex  = db.load_gex_surface("SPY", "2025-06-09", "2025-06-10")
    snap = db.load_greeks_snapshot("SPY", "2025-06-10 15:30:00")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import duckdb as _duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False


class MicrostructureDB:
    """
    Unified read interface for the microstructure data lake using DuckDB.

    Parameters
    ----------
    data_path : Root directory of the microstructure lake (default: ``data/microstructure``).
    db_file   : Optional persistent DuckDB file.  Pass ``None`` (default) for
                an in-memory connection (recommended for analytics; faster startup).
    """

    def __init__(
        self,
        data_path: str = "data/microstructure",
        *,
        db_file: str | None = None,
    ) -> None:
        if not _HAS_DUCKDB:
            raise ImportError(
                "DuckDB is required for MicrostructureDB.\n"
                "Install it with:  pip install 'regime-locus-matrix[microstructure]'"
            )
        self.data_path = Path(data_path)
        self._db_file = db_file
        self._conn: "_duckdb.DuckDBPyConnection | None" = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> "_duckdb.DuckDBPyConnection":
        """Return (and lazily create) the DuckDB connection."""
        if self._conn is None:
            if self._db_file:
                self._conn = _duckdb.connect(self._db_file)
            else:
                self._conn = _duckdb.connect()
        return self._conn

    def close(self) -> None:
        """Close the connection if open."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "MicrostructureDB":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def _query(self, sql: str) -> Generator[pd.DataFrame, None, None]:
        """Execute *sql* and yield the result DataFrame; propagates exceptions."""
        conn = self.connect()
        try:
            yield conn.execute(sql).fetchdf()
        except Exception as exc:
            logger.error("MicrostructureDB query failed:\n%s\nError: %s", sql, exc)
            raise

    def _glob(self, *parts: str) -> str:
        """Build a quoted glob path for DuckDB."""
        return str(self.data_path.joinpath(*parts) / "*.parquet")

    # ------------------------------------------------------------------
    # Underlying 1-second / 5-second bars
    # ------------------------------------------------------------------

    def load_underlying_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        bar_resolution: str = "1s",
    ) -> pd.DataFrame:
        """
        Load underlying OHLCV bars between two dates (inclusive).

        Parameters
        ----------
        symbol          : Ticker (e.g. "SPY")
        start_date      : ISO date string "YYYY-MM-DD"
        end_date        : ISO date string "YYYY-MM-DD"
        bar_resolution  : Sub-directory under the symbol lake ("1s", "5s", "1m")

        Returns
        -------
        DataFrame with columns: timestamp, open, high, low, close, volume, vwap
        """
        glob = self._glob(f"underlying/{symbol}/{bar_resolution}")
        sql = f"""
            SELECT *
            FROM '{glob}'
            WHERE timestamp BETWEEN TIMESTAMP '{start_date} 00:00:00'
                                AND TIMESTAMP '{end_date} 23:59:59'
            ORDER BY timestamp
        """
        with self._query(sql) as df:
            return df

    # ------------------------------------------------------------------
    # Greeks snapshots
    # ------------------------------------------------------------------

    def load_greeks_snapshot(
        self,
        symbol: str,
        timestamp: str,
    ) -> pd.DataFrame:
        """Load all option contract Greeks at a specific snapshot timestamp."""
        glob = self._glob(f"options/{symbol}/greeks_snapshots")
        sql = f"""
            SELECT *
            FROM '{glob}'
            WHERE underlying_symbol = '{symbol}'
              AND timestamp = TIMESTAMP '{timestamp}'
        """
        with self._query(sql) as df:
            return df

    def load_greeks_range(
        self,
        symbol: str,
        start_ts: str,
        end_ts: str,
        *,
        strikes: list[float] | None = None,
        max_dte: int | None = None,
    ) -> pd.DataFrame:
        """
        Load greeks snapshots between two timestamps.

        Optional filters on strike list and maximum days-to-expiry reduce I/O.
        """
        glob = self._glob(f"options/{symbol}/greeks_snapshots")
        conditions = [
            f"underlying_symbol = '{symbol}'",
            f"timestamp BETWEEN TIMESTAMP '{start_ts}' AND TIMESTAMP '{end_ts}'",
        ]
        if strikes:
            strike_list = ", ".join(str(k) for k in strikes)
            conditions.append(f"strike IN ({strike_list})")
        if max_dte is not None:
            conditions.append(f"dte <= {max_dte}")

        sql = f"""
            SELECT *
            FROM '{glob}'
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp, strike, expiration
        """
        with self._query(sql) as df:
            return df

    # ------------------------------------------------------------------
    # GEX surface
    # ------------------------------------------------------------------

    def load_gex_surface(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        net_gex_only: bool = False,
    ) -> pd.DataFrame:
        """Load pre-computed GEX surface rows between two dates."""
        glob = self._glob(f"options/{symbol}/derived/gex_surface")
        select = "timestamp, underlying_symbol, underlying_price, strike, expiration, gex, net_gex"
        if not net_gex_only:
            select = "*"
        sql = f"""
            SELECT {select}
            FROM '{glob}'
            WHERE underlying_symbol = '{symbol}'
              AND timestamp BETWEEN TIMESTAMP '{start_date} 00:00:00'
                                AND TIMESTAMP '{end_date} 23:59:59'
            ORDER BY timestamp, strike
        """
        with self._query(sql) as df:
            return df

    def load_gex_flip_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Return a time series of net-GEX sign changes (the GEX flip events).

        Result columns: timestamp, prev_net_gex, net_gex, flip_direction
        """
        gex = self.load_gex_surface(symbol, start_date, end_date, net_gex_only=True)
        if gex.empty:
            return pd.DataFrame()

        by_time = gex.groupby("timestamp")["net_gex"].sum().reset_index()
        by_time = by_time.sort_values("timestamp").reset_index(drop=True)
        by_time["prev_net_gex"] = by_time["net_gex"].shift(1)
        flips = by_time[
            (by_time["prev_net_gex"] * by_time["net_gex"] < 0)
            & by_time["prev_net_gex"].notna()
        ].copy()
        flips["flip_direction"] = flips.apply(
            lambda r: "positive_to_negative" if r["prev_net_gex"] > 0 else "negative_to_positive",
            axis=1,
        )
        return flips[["timestamp", "prev_net_gex", "net_gex", "flip_direction"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # IV surface
    # ------------------------------------------------------------------

    def load_iv_surface(self, symbol: str, date: str) -> pd.DataFrame:
        """Load the pre-built IV surface for a specific calendar date."""
        file_path = self.data_path / f"options/{symbol}/derived/iv_surface/{symbol}_{date}.parquet"
        if not file_path.exists():
            logger.warning("IV surface file not found: %s", file_path)
            return pd.DataFrame()
        sql = f"SELECT * FROM '{file_path}'"
        with self._query(sql) as df:
            return df

    def load_iv_surface_range(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load IV surfaces for a range of dates."""
        glob = self._glob(f"options/{symbol}/derived/iv_surface")
        sql = f"""
            SELECT *
            FROM '{glob}'
            WHERE underlying_symbol = '{symbol}'
              AND timestamp BETWEEN TIMESTAMP '{start_date} 00:00:00'
                                AND TIMESTAMP '{end_date} 23:59:59'
            ORDER BY timestamp, days_to_expiry, moneyness
        """
        with self._query(sql) as df:
            return df

    # ------------------------------------------------------------------
    # Convenience: regime context for a timestamp
    # ------------------------------------------------------------------

    def microstructure_regime_context(
        self,
        symbol: str,
        timestamp: str,
        *,
        dte_bucket: float = 30.0,
    ) -> dict[str, float]:
        """
        Return headline microstructure regime signals for a specific timestamp.

        Suitable for injecting into the RLM FactorPipeline as supplementary context.

        Returns
        -------
        dict with keys:
            gex_net_total, gex_flip_strike, iv_atm, iv_skew_25d, iv_term_ratio
        """
        from rlm.microstructure.calculators.gex import aggregate_gex_profile
        from rlm.microstructure.calculators.iv_surface import (
            build_iv_surface,
            query_iv_surface,
            skew_at_dte,
        )

        date = str(timestamp)[:10]
        snapshot = self.load_greeks_snapshot(symbol, timestamp)

        ctx: dict[str, float] = {
            "gex_net_total": float("nan"),
            "gex_flip_strike": float("nan"),
            "iv_atm": float("nan"),
            "iv_skew_25d": float("nan"),
            "iv_term_ratio": float("nan"),
        }
        if snapshot.empty:
            return ctx

        from rlm.microstructure.calculators.gex import build_gex_surface_from_df
        gex_df = build_gex_surface_from_df(snapshot, underlying_symbol=symbol, timestamp=timestamp)
        profile = aggregate_gex_profile(gex_df)
        ctx["gex_net_total"] = profile["total_net_gex"]
        ctx["gex_flip_strike"] = profile["gex_flip_strike"]

        iv_surface = build_iv_surface(snapshot, timestamp=timestamp, underlying_symbol=symbol)
        if not iv_surface.empty:
            ctx["iv_atm"] = query_iv_surface(iv_surface, moneyness=1.0, dte=dte_bucket)
            ctx["iv_skew_25d"] = skew_at_dte(iv_surface, dte=dte_bucket)
            iv_short = query_iv_surface(iv_surface, moneyness=1.0, dte=7.0)
            iv_long = query_iv_surface(iv_surface, moneyness=1.0, dte=90.0)
            if iv_long and iv_long > 0:
                ctx["iv_term_ratio"] = iv_short / iv_long

        return ctx

    # ------------------------------------------------------------------
    # Raw SQL passthrough
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary DuckDB SQL and return a DataFrame."""
        with self._query(sql) as df:
            return df
