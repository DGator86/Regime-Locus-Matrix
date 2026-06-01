"""Rolling incremental bar store.

Keeps ``data/raw/bars_{SYMBOL}.csv`` current by fetching only the bars that
are missing since the last stored timestamp, then appending, deduplicating,
and sorting before writing back.

Usage::

    from rlm.data.rolling_store import RollingBarsStore

    store = RollingBarsStore("SPY")
    result = store.update()          # fetch & append any new bars
    print(result.new_rows, result.total_rows, result.last_timestamp)

    # Or from a custom data root:
    store = RollingBarsStore("SPY", data_root="/mnt/lake")
    result = store.update()

CLI equivalent::

    python -m rlm.data.rolling_store SPY
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from rlm.data.paths import get_raw_data_dir

_log = logging.getLogger(__name__)

_DATE_FORMAT = "%Y-%m-%d"
_OVERLAP_DAYS = 5  # re-fetch a small tail to catch late-arriving / corrected bars


@dataclass
class RollingUpdateResult:
    symbol: str
    new_rows: int
    total_rows: int
    last_timestamp: pd.Timestamp | None
    already_current: bool
    bars_path: Path


class RollingBarsStore:
    """Incrementally keep a local CSV bars file up to date.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    data_root:
        Override for the data root directory.  Defaults to ``RLM_DATA_ROOT``
        env var or ``cwd/data``.
    interval:
        yfinance interval string (default ``"1d"``).
    lookback_days:
        Number of calendar days to fetch on first run (no existing file).
    """

    def __init__(
        self,
        symbol: str,
        data_root: str | Path | None = None,
        interval: str = "1d",
        lookback_days: int = 730,
    ) -> None:
        self.symbol = symbol.upper().strip()
        self.interval = interval
        self.lookback_days = lookback_days
        self._raw_dir = get_raw_data_dir(data_root)
        self._bars_path = self._raw_dir / f"bars_{self.symbol}.csv"

    @property
    def bars_path(self) -> Path:
        return self._bars_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame | None:
        """Return the current stored bars, or ``None`` if no file exists."""
        if not self._bars_path.is_file():
            return None
        df = pd.read_csv(self._bars_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def save(self, df: pd.DataFrame) -> None:
        """Replace stored bars with ``df`` (sorted, deduped by timestamp)."""
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.normalize()
        out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
        self._write_csv_atomic(out.reset_index(drop=True))

    def update(self, *, today: date | None = None) -> RollingUpdateResult:
        """Fetch any bars newer than the last stored timestamp and append them.

        Returns a :class:`RollingUpdateResult` describing what changed.
        """
        today = today or date.today()
        existing = self.load()

        if existing is not None and not existing.empty:
            last_ts = pd.Timestamp(existing["timestamp"].max()).date()
            if last_ts >= today:
                return RollingUpdateResult(
                    symbol=self.symbol,
                    new_rows=0,
                    total_rows=len(existing),
                    last_timestamp=pd.Timestamp(last_ts),
                    already_current=True,
                    bars_path=self._bars_path,
                )
            fetch_start = last_ts - timedelta(days=_OVERLAP_DAYS)
        else:
            fetch_start = today - timedelta(days=self.lookback_days)

        fetch_end = today + timedelta(days=1)  # yfinance end is exclusive

        _log.info(
            "RollingBarsStore[%s]: fetching %s → %s (interval=%s)",
            self.symbol,
            fetch_start.strftime(_DATE_FORMAT),
            today.strftime(_DATE_FORMAT),
            self.interval,
        )

        fresh = self._fetch(fetch_start, fetch_end)
        if fresh.empty:
            _log.warning("RollingBarsStore[%s]: provider returned no new bars", self.symbol)
            total = len(existing) if existing is not None else 0
            last = pd.Timestamp(existing["timestamp"].max()) if existing is not None and not existing.empty else None
            return RollingUpdateResult(
                symbol=self.symbol,
                new_rows=0,
                total_rows=total,
                last_timestamp=last,
                already_current=False,
                bars_path=self._bars_path,
            )

        merged = self._merge(existing, fresh)
        new_rows = len(merged) - (len(existing) if existing is not None else 0)

        self._write_csv_atomic(merged)

        _log.info(
            "RollingBarsStore[%s]: +%d new rows → %d total, saved to %s",
            self.symbol,
            max(new_rows, 0),
            len(merged),
            self._bars_path,
        )

        return RollingUpdateResult(
            symbol=self.symbol,
            new_rows=max(new_rows, 0),
            total_rows=len(merged),
            last_timestamp=pd.Timestamp(merged["timestamp"].max()),
            already_current=False,
            bars_path=self._bars_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("yfinance is required for RollingBarsStore") from exc

        ticker = yf.Ticker(self.symbol)
        raw = ticker.history(
            start=start.strftime(_DATE_FORMAT),
            end=end.strftime(_DATE_FORMAT),
            interval=self.interval,
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw.index.name = "timestamp"
        raw.columns = [str(c).lower() for c in raw.columns]
        df = raw.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        return df

    @staticmethod
    def _merge(existing: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
        existing_norm = None
        if existing is not None and not existing.empty:
            existing_norm = existing.copy()
            existing_norm["timestamp"] = pd.to_datetime(existing_norm["timestamp"]).dt.normalize()

        fresh_norm = fresh.copy()
        fresh_norm["timestamp"] = pd.to_datetime(fresh_norm["timestamp"]).dt.normalize()

        parts = [existing_norm, fresh_norm] if existing_norm is not None else [fresh_norm]
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")

        if existing_norm is not None:
            missing_fresh_cols = [
                col
                for col in existing_norm.columns
                if col != "timestamp" and col not in fresh_norm.columns and col in combined.columns
            ]
            if missing_fresh_cols:
                existing_by_timestamp = existing_norm.drop_duplicates(subset=["timestamp"], keep="last").set_index(
                    "timestamp"
                )
                combined_by_timestamp = combined.set_index("timestamp")
                for col in missing_fresh_cols:
                    combined_by_timestamp[col] = combined_by_timestamp[col].fillna(existing_by_timestamp[col])
                combined = combined_by_timestamp.reset_index()

        return combined.sort_values("timestamp").reset_index(drop=True)

    def _write_csv_atomic(self, df: pd.DataFrame) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._bars_path.name}.",
            suffix=".tmp",
            dir=self._raw_dir,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", newline="") as handle:
                df.to_csv(handle, index=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._bars_path)
            self._fsync_dir()
        except Exception:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _fsync_dir(self) -> None:
        try:
            dir_fd = os.open(self._raw_dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    symbols = sys.argv[1:] or ["SPY"]
    for sym in symbols:
        result = RollingBarsStore(sym).update()
        status = "already current" if result.already_current else f"+{result.new_rows} new rows"
        print(
            f"{result.symbol}: {status} | total={result.total_rows} | "
            f"last={result.last_timestamp.date() if result.last_timestamp else 'N/A'}"
        )
