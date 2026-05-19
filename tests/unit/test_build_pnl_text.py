from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlm.data.ibkr_snapshot import IbkrSnapshot  # noqa: E402
from rlm.notify.telegram_rlm import build_pnl_text  # noqa: E402


def _stub_ibkr(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = IbkrSnapshot(positions=(), account_summary=(), host="127.0.0.1", port=4002, client_id=1)

    def _fetch(**_kwargs: object) -> IbkrSnapshot:
        return empty

    import rlm.data.ibkr_snapshot as snap_mod

    monkeypatch.setattr(snap_mod, "fetch_ibkr_account_snapshot", _fetch)


def test_build_pnl_text_options_open_mtm_consistent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_ibkr(monkeypatch)
    monkeypatch.setenv("RLM_LARGE_OPTIONS_BOOK_SEED", "250000")
    monkeypatch.setenv("RLM_LARGE_EQUITY_BOOK_SEED", "250000")
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    (proc / "trade_log.csv").write_text(
        "plan_id,symbol,timestamp_utc,closed,unrealized_pnl,action,quantity\n"
        "eq1,SPY,2026-01-02T15:00:00+00:00,0,59,BUY,1\n"
        "opt1,QQQ,2026-01-02T16:00:00+00:00,0,-2506,,\n",
        encoding="utf-8",
    )
    (proc / "equity_trade_log.csv").write_text(
        "plan_id,symbol,timestamp_utc,action,quantity,unrealized_pnl,signal,closed\n",
        encoding="utf-8",
    )
    ch = tmp_path / "data" / "challenge"
    ch.mkdir(parents=True)
    (ch / "state.json").write_text(
        '{"balance":1000,"seed":1000,"target":25000,"trade_history":[],"open_positions":[]}',
        encoding="utf-8",
    )

    text = build_pnl_text(tmp_path)
    assert "Open MTM (unrealized):           $-2,506.00" in text
    assert "open MTM $-2,506.00)" in text
    assert "$-2,565.00" not in text


@pytest.mark.parametrize(
    "line",
    [
        "Realized all-time (closed only): $0.00",
        "Open MTM (unrealized):           $0.00",
    ],
)
def test_build_pnl_equities_section_flat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, line: str
) -> None:
    _stub_ibkr(monkeypatch)
    monkeypatch.setenv("RLM_LARGE_EQUITY_BOOK_SEED", "250000")
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    (proc / "equity_trade_log.csv").write_text(
        "plan_id,symbol,timestamp_utc,action,quantity,unrealized_pnl,signal,closed\n",
        encoding="utf-8",
    )
    (proc / "trade_log.csv").write_text("plan_id,symbol,timestamp_utc,closed,unrealized_pnl\n", encoding="utf-8")
    ch = tmp_path / "data" / "challenge"
    ch.mkdir(parents=True)
    (ch / "state.json").write_text(
        '{"balance":1000,"seed":1000,"target":25000,"trade_history":[],"open_positions":[]}',
        encoding="utf-8",
    )
    text = build_pnl_text(tmp_path)
    assert line in text
