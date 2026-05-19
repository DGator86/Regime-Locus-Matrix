from __future__ import annotations

from rlm.data.ibkr_snapshot import (
    IbkrAccountSummaryRow,
    account_summary_tag_float,
    format_account_summary_money,
)


def test_account_summary_prefers_usd_over_other_ccy() -> None:
    summary = (
        IbkrAccountSummaryRow("DU123", "NetLiquidation", "100", "EUR"),
        IbkrAccountSummaryRow("DU123", "NetLiquidation", "250000", "USD"),
    )
    assert account_summary_tag_float(summary, "NetLiquidation") == 250_000.0


def test_account_summary_zero_is_valid() -> None:
    summary = (IbkrAccountSummaryRow("DU123", "UnrealizedPnL", "0", "USD"),)
    assert account_summary_tag_float(summary, "UnrealizedPnL") == 0.0
    assert format_account_summary_money(summary, "UnrealizedPnL") == "$0.00"


def test_account_summary_missing_returns_none() -> None:
    summary = (IbkrAccountSummaryRow("DU123", "NetLiquidation", "1", "USD"),)
    assert account_summary_tag_float(summary, "RealizedPnL") is None
    assert format_account_summary_money(summary, "RealizedPnL") == "—"
