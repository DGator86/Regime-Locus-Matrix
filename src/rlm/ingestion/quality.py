from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class QualityResult:
    ok: bool
    reasons: list[str]


def validate_bar_timestamps(df: pd.DataFrame, *, ts_col: str = "timestamp") -> QualityResult:
    if df.empty:
        return QualityResult(ok=False, reasons=["empty-bars"])
    if ts_col not in df.columns:
        return QualityResult(ok=False, reasons=[f"missing-{ts_col}"])
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    reasons: list[str] = []
    if ts.isna().any():
        reasons.append("invalid-timestamp")
    if ts.duplicated().any():
        reasons.append("duplicate-timestamp")
    if ts.sort_values().diff().dropna().le(pd.Timedelta(0)).any():
        reasons.append("non-monotonic-time")
    return QualityResult(ok=not reasons, reasons=reasons)


def validate_option_chain(df: pd.DataFrame) -> QualityResult:
    required = {"strike", "expiration", "bid", "ask", "option_type"}
    missing = sorted(required - set(df.columns))
    reasons: list[str] = [f"missing-col:{m}" for m in missing]
    if not missing:
        if (pd.to_numeric(df["bid"], errors="coerce") < 0).any() or (pd.to_numeric(df["ask"], errors="coerce") < 0).any():
            reasons.append("negative-bid-ask")
        if pd.to_datetime(df["expiration"], errors="coerce").isna().any():
            reasons.append("invalid-expiration")
        if df.groupby(["expiration", "option_type"])["strike"].nunique().min() < 3:
            reasons.append("incomplete-strike-ladder")
    return QualityResult(ok=not reasons, reasons=reasons)


def validate_option_contracts(df: pd.DataFrame) -> QualityResult:
    """Validate Massive reference option contract rows before writing them."""
    if df.empty:
        return QualityResult(ok=False, reasons=["empty-contracts"])

    required = {"ticker", "underlying_ticker", "strike_price", "expiration_date", "contract_type"}
    missing = sorted(required - set(df.columns))
    reasons: list[str] = [f"missing-col:{m}" for m in missing]
    if missing:
        return QualityResult(ok=False, reasons=reasons)

    strikes = pd.to_numeric(df["strike_price"], errors="coerce")
    if strikes.isna().any() or strikes.le(0).any():
        reasons.append("invalid-strike-price")

    expirations = pd.to_datetime(df["expiration_date"], errors="coerce")
    if expirations.isna().any():
        reasons.append("invalid-expiration-date")

    contract_types = df["contract_type"].astype(str).str.lower().str.strip()
    if (~contract_types.isin(["call", "put"])).any():
        reasons.append("invalid-contract-type")

    if df["ticker"].isna().any() or df["ticker"].astype(str).str.strip().eq("").any():
        reasons.append("missing-ticker")

    if df["underlying_ticker"].isna().any() or df["underlying_ticker"].astype(str).str.strip().eq("").any():
        reasons.append("missing-underlying-ticker")

    return QualityResult(ok=not reasons, reasons=reasons)
