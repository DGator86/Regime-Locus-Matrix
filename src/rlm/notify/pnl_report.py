"""
EOD PnL Report — daily performance across the books you run in parallel.

**Options (swing / large-account universe monitor)** — ``data/processed/trade_log.csv``:
per-poll append, ``unrealized_pnl`` = mark minus entry debit. The headline
“win%” is mark quality, not closed-round-trip quality; we split open vs
exit rows.

**Equities (regime stock leg)** — ``data/processed/equity_trade_log.csv``: same
shape idea (``action``/``quantity`` present); one section when the file exists.

**$1K → $25K PDT options challenge (dry-run)** — ``data/challenge/state.json`` (and
optional ``data/challenge/trade_log.csv`` for history): session closed P&L
from ``trade_history`` with ``exit_date`` on the report day, plus open
``unrealised_pnl`` from the state file.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

_CH_TITLE = "Challenge $1K→$25K (PDT / dry-run)"
_EQUITY_TITLE = "Equities (IBKR regime log)"
_OPT_TITLE = "Options (universe monitor / swing or large acct)"


def _now_utc() -> datetime:
    """Test hook: patch this instead of :func:`datetime.now`."""
    return datetime.now(timezone.utc)


def _row_time_utc(iso_ts: str) -> datetime | None:
    raw = (iso_ts or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _is_equity_log_row(r: dict[str, str]) -> bool:
    """``ibkr_equity_paper_trade`` log rows (see ``show_pnl.py``)."""
    c = set(r.keys())
    return "action" in c and "quantity" in c


def _rows_for_session_day(
    all_rows: list[dict[str, str]], session_date, *, equity_only: bool | None = None
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in all_rows:
        if equity_only is True and not _is_equity_log_row(r):
            continue
        if equity_only is False and _is_equity_log_row(r):
            continue
        ts = _row_time_utc(r.get("timestamp_utc", ""))
        if ts is None:
            continue
        if ts.astimezone(_ET).date() == session_date:
            out.append(r)
    return out


def _latest_per_plan(today_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for r in today_rows:
        pid = str(r.get("plan_id", ""))
        if pid:
            latest[pid] = r
    return latest


def _pnl(r: dict[str, str]) -> float:
    try:
        return float(r.get("unrealized_pnl", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_closed(r: dict[str, str]) -> bool:
    return str(r.get("closed", "0")).strip() == "1"


def _format_log_section(
    *,
    title: str,
    today_rows: list[dict[str, str]],
) -> str:
    """Block for one options-style or equity-style day slice."""
    if not today_rows:
        return f"<b>{title}</b>\n  (no log rows for this session)\n"

    latest = _latest_per_plan(today_rows)
    pnl_sum = 0.0
    open_mtm = 0.0
    open_n = 0
    o_win = o_loss = 0
    closed_pnl = 0.0
    closed_n = 0
    c_win = c_loss = 0
    c_win_sum = 0.0
    c_loss_sum = 0.0
    wins = 0
    sym_pnl: dict[str, float] = {}
    n_plan = 0
    for _pid, r in latest.items():
        p = _pnl(r)
        pnl_sum += p
        n_plan += 1
        sym = str(r.get("symbol", "") or "?")
        sym_pnl[sym] = sym_pnl.get(sym, 0.0) + p
        if p > 0:
            wins += 1
        if _is_closed(r):
            closed_pnl += p
            closed_n += 1
            if p > 0:
                c_win += 1
                c_win_sum += p
            elif p < 0:
                c_loss += 1
                c_loss_sum += abs(p)
        else:
            open_mtm += p
            open_n += 1
            if p > 0:
                o_win += 1
            elif p < 0:
                o_loss += 1
    wr = (wins / n_plan * 100) if n_plan else 0.0
    c_wr = (c_win / closed_n * 100) if closed_n else 0.0
    worst: list[str] = []
    for s, v in sorted(sym_pnl.items(), key=lambda x: x[1])[:5]:
        if v < 0:
            worst.append(f"{s} {v:+.2f}")
    worst_line = f"  Worst MTM (symbol sum): {', '.join(worst)}\n" if worst else ""
    mtm_note = (
        f"  (headline MTM win% {wr:.1f}% = mark vs entry, not round-trip; " f"use exit line for stop/TP quality.)\n"
    )
    if n_plan <= 30:
        mtm_note = "\n"  # preserve newline so exits line doesn't merge with next
    payoff_line = ""
    if closed_n >= 2 and c_win >= 1 and c_loss >= 1:
        avg_win = c_win_sum / c_win
        avg_loss = c_loss_sum / c_loss
        ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        min_ratio = (1.0 - c_wr / 100.0) / (c_wr / 100.0) if c_wr > 0 else float("inf")
        payoff_line = (
            f"  Exit payoff: avg win ${avg_win:.0f} / avg loss ${avg_loss:.0f} = "
            f"{ratio:.2f}x  (min +EV at {c_wr:.0f}% wr: {min_ratio:.2f}x)\n"
        )
    conc_line = ""
    if pnl_sum < 0 and sym_pnl:
        worst_sym, worst_val = min(sym_pnl.items(), key=lambda x: x[1])
        if worst_val < 0:
            conc_pct = abs(worst_val) / abs(pnl_sum) * 100
            if conc_pct >= 40:
                conc_line = f"  Concentration: {worst_sym} = {conc_pct:.0f}% of session PnL\n"
    return (
        f"<b>{title}</b>\n"
        f"  Mark vs entry (last row/plan, session): <b>${pnl_sum:+.2f}</b>\n"
        f"  Open: {open_n}  MTM ${open_mtm:+.2f}  (+{o_win} / −{o_loss} vs entry)\n"
        f"  Exits (closed=1): {closed_n}  at-exit P&L ${closed_pnl:+.2f}  "
        f"{c_win}W / {c_loss}L  ({c_wr:.1f}% green){mtm_note}"
        f"  unique plan_id: {n_plan}\n"
        f"{worst_line}"
        f"{payoff_line}"
        f"{conc_line}"
    )


def _format_challenge_eod(root: Path, session_date: date_type) -> str | None:
    state_path = root / "data" / "challenge" / "state.json"
    if not state_path.is_file():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"<b>{_CH_TITLE}</b>\n  (state.json present but not readable)\n"
    bal = float(data.get("balance", 0.0))
    seed = float(data.get("seed", 1_000.0))
    target = float(data.get("target", 25_000.0))
    progress = 0.0
    if target > seed:
        progress = min(1.0, max(0.0, (bal - seed) / (target - seed)))
    o_pos = data.get("open_positions") or []
    if not isinstance(o_pos, list):
        o_pos = []
    open_mtm = sum(float(p.get("unrealised_pnl", 0.0) or 0.0) for p in o_pos)  # type: ignore[misc]
    n_open = len([p for p in o_pos if str(p.get("status", "open")) == "open"])  # type: ignore[misc]
    sd = str(session_date)
    th_raw = data.get("trade_history")
    th: list = th_raw if isinstance(th_raw, list) else []
    day_closed: list[dict] = []
    for t in th:
        ex = str(t.get("exit_date", ""))[:10]
        if ex == sd:
            day_closed.append(t)
    d_pnl = 0.0
    d_w = 0
    d_l = 0
    for t in day_closed:
        try:
            p = float(t.get("pnl", 0.0) or 0.0)
        except (TypeError, ValueError):
            p = 0.0
        d_pnl += p
        if p > 0:
            d_w += 1
        else:
            d_l += 1
    w_all = sum(1 for x in th if float(x.get("pnl", 0) or 0) > 0)  # type: ignore[misc, arg-type]
    n_all = len(th)
    book_wr = 100.0 * w_all / n_all if n_all else 0.0
    return (
        f"<b>{_CH_TITLE}</b>\n"
        f"  Balance: ${bal:,.2f}  target ${target:,.0f}  path {progress*100:.1f}%  "
        f"open {n_open}  uPnL ${open_mtm:+.2f}\n"
        f"  <b>Today closed (exit_date = session):</b> ${d_pnl:+.2f}  {d_w}W / {d_l}L  "
        f"({len(day_closed)} round-trip(s))\n"
        f"  All-time closed: {n_all} trades, wins {w_all}  (book ≈ {book_wr:.1f}% win count)\n"
    )


def _load_all_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    return list(_iter_csv_rows(path))


def calculate_daily_pnl(root: Path) -> str:
    """Build HTML-ish Telegram string for the session (US/Eastern *calendar* day)."""
    now_utc = _now_utc()
    session_date = now_utc.astimezone(_ET).date()

    try:
        blocks: list[str] = [
            f"<b>[EOD Report] {session_date} (ET)</b>\n"
            f"<i>Options = swing/large acct log · Equities = stock leg · "
            f"Challenge = PDT $1K→$25K dry-run</i>\n",
        ]

        opt_path = root / "data" / "processed" / "trade_log.csv"
        if opt_path.is_file():
            raw = _load_all_rows(opt_path)
            if not raw:
                blocks.append(f"<b>{_OPT_TITLE}</b>\n  (file empty)\n")
            else:
                today = _rows_for_session_day(raw, session_date, equity_only=False)
                today_opt = [r for r in today if not _is_equity_log_row(r)]
                blocks.append(_format_log_section(title=_OPT_TITLE, today_rows=today_opt))
        else:
            msg = f"<b>{_OPT_TITLE}</b>\n  (no options trade_log — run universe monitor)\n"
            blocks.append(msg)

        eq_path = root / "data" / "processed" / "equity_trade_log.csv"
        if eq_path.is_file():
            eq_raw = _load_all_rows(eq_path)
            if not eq_raw:
                blocks.append(f"<b>{_EQUITY_TITLE}</b>\n  (file empty)\n")
            else:
                today_eq = _rows_for_session_day(eq_raw, session_date, equity_only=True)
                blocks.append(_format_log_section(title=_EQUITY_TITLE, today_rows=today_eq))
        else:
            blocks.append(f"<b>{_EQUITY_TITLE}</b>\n  (no file — run ibkr_equity_paper_trade to log this)\n")

        ch = _format_challenge_eod(root, session_date)
        if ch is not None:
            blocks.append(ch)
        else:
            blocks.append(f"<b>{_CH_TITLE}</b>\n  (no data/challenge/state.json — run rlm challenge)\n")

        return "\n".join(blocks)
    except Exception as e:  # noqa: BLE001
        return f"Error generating PnL report: {e}"
