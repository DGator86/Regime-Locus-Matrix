from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rlm.data.lake.writers import save_parquet_versioned
from rlm.ingestion.quality import validate_bar_timestamps, validate_option_chain, validate_option_contracts
from rlm.ingestion.writers import write_massive_option_contracts_parquet


def test_validate_bar_timestamps_rejects_duplicates() -> None:
    df = pd.DataFrame({"timestamp": ["2025-01-01T10:00:00Z", "2025-01-01T10:00:00Z"]})
    res = validate_bar_timestamps(df)
    assert not res.ok
    assert "duplicate-timestamp" in res.reasons


def test_validate_option_chain_rejects_negative_bid_ask() -> None:
    df = pd.DataFrame(
        {
            "strike": [100, 105, 110],
            "expiration": ["2025-02-21", "2025-02-21", "2025-02-21"],
            "bid": [1.0, -1.0, 0.5],
            "ask": [1.1, 1.2, 0.6],
            "option_type": ["call", "call", "call"],
        }
    )
    res = validate_option_chain(df)
    assert not res.ok
    assert "negative-bid-ask" in res.reasons


def test_validate_option_contracts_accepts_massive_reference_rows() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["O:SPY250221C00500000", "O:SPY250221P00500000"],
            "underlying_ticker": ["SPY", "SPY"],
            "contract_type": ["call", "put"],
            "expiration_date": ["2025-02-21", "2025-02-21"],
            "strike_price": [500.0, 500.0],
        }
    )
    res = validate_option_contracts(df)
    assert res.ok


def test_write_massive_option_contracts_uses_contract_quality_gate(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "ticker": ["O:SPY250221C00500000", "O:SPY250221P00500000"],
            "underlying_ticker": ["SPY", "SPY"],
            "contract_type": ["call", "put"],
            "expiration_date": ["2025-02-21", "2025-02-21"],
            "strike_price": [500.0, 500.0],
        }
    )
    monkeypatch.setattr("rlm.ingestion.writers.MassiveContractsFetcher", lambda: _StubContractsFetcher(df))
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False: Path(path).write_text("ok", encoding="utf-8"),
    )

    out = write_massive_option_contracts_parquet("SPY", repo_root=tmp_path)

    assert out == tmp_path / "data" / "options" / "SPY" / "contracts" / "spy_all_contracts.parquet"
    assert out.exists()


def test_save_parquet_versioned_writes_lineage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False: Path(path).write_text("ok", encoding="utf-8"),
    )
    p = tmp_path / "data" / "stocks" / "SPY" / "1d" / "spy.parquet"
    df = pd.DataFrame({"timestamp": ["2025-01-01"], "open": [1.0]})
    save_parquet_versioned(df, p, source="test", schema="bars", lineage_root=tmp_path / "data")
    log = tmp_path / "data" / "metadata" / "lineage" / "lineage_log.jsonl"
    assert log.exists()
    rec = json.loads(log.read_text(encoding="utf-8").strip())
    assert rec["schema"] == "bars"


class _StubContractsFetcher:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def fetch(self, underlying: str, **params) -> pd.DataFrame:
        return self.df
