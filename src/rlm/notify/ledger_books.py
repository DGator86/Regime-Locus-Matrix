"""Seeded book accounting + CSV “spreadsheet” exports for parallel trading books.

Three sleeves:
- **Large equities** (IBKR prop-style log) — notional seed via ``RLM_LARGE_EQUITY_BOOK_SEED``.
- **Large options** (universe monitor CSV) — notional seed via ``RLM_LARGE_OPTIONS_BOOK_SEED``.
- **PDT / Robinhood challenge** — seed from ``data/challenge/state.json`` (default $1k→$25k).

Ledgers are UTF-8 CSV files under ``data/processed/ledgers/`` for Excel / Google Sheets import.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rlm.notify.options_paths import options_trade_log_read_paths

_LEDGER_COLS = (
    "book",
    "event",
    "plan_id",
    "symbol",
    "timestamp_utc",
    "pnl_usd",
    "running_closed_usd",
    "running_book_usd",
    "detail",
)


def large_equity_book_seed() -> float:
    raw = (os.environ.get("RLM_LARGE_EQUITY_BOOK_SEED") or "250000").strip()
    return max(0.0, float(raw))


def large_options_book_seed() -> float:
    raw = (os.environ.get("RLM_LARGE_OPTIONS_BOOK_SEED") or "250000").strip()
    return max(0.0, float(raw))


def ledger_dir(root: Path) -> Path:
    d = root / "data" / "processed" / "ledgers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _iter_csv(path: Path) -> Iterator[dict[str, str]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _is_equity_row(r: dict[str, str]) -> bool:
    return "action" in r and "quantity" in r


def _parse_ts(raw: str) -> datetime | None:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _options_closed_open_maps(
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    closed_by_pid: dict[str, dict[str, str]] = {}
    latest_by_pid: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = str(row.get("plan_id") or "")
        if not pid or _is_equity_row(row):
            continue
        latest_by_pid[pid] = row
        if (row.get("closed") or "0").strip() == "1":
            closed_by_pid[pid] = row
    return closed_by_pid, latest_by_pid


def _equity_closed_open_maps(
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    closed_by_pid: dict[str, dict[str, str]] = {}
    latest_by_pid: dict[str, dict[str, str]] = {}
    for row in rows:
        if not _is_equity_row(row):
            continue
        pid = str(row.get("plan_id") or "")
        if not pid:
            continue
        latest_by_pid[pid] = row
        if (row.get("closed") or "0").strip() == "1":
            closed_by_pid[pid] = row
    return closed_by_pid, latest_by_pid


def _float_cell(r: dict[str, str], key: str) -> float:
    try:
        return float(r.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _write_ledger(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "# RLM trading ledger — import into Excel or Google Sheets.",
                f"generated_utc={datetime.now(tz=timezone.utc).isoformat()}",
            ]
        )
        w.writerow(_LEDGER_COLS)
        for row in rows:
            w.writerow(row)


@dataclass(frozen=True)
class BookSnapshot:
    seed: float
    closed_realized: float
    open_mtm: float

    @property
    def book_value(self) -> float:
        return self.seed + self.closed_realized + self.open_mtm


def _load_options_rows(root: Path) -> list[dict[str, str]]:
    for cand in options_trade_log_read_paths(root):
        if cand.is_file():
            rows = list(_iter_csv(cand))
            if rows:
                return rows
    return []


def _load_equity_rows(root: Path) -> list[dict[str, str]]:
    raw = (os.environ.get("RLM_EQUITY_TRADE_LOG_PATH") or "").strip()
    if raw:
        p = Path(raw)
        path = p if p.is_absolute() else root / p
    else:
        path = root / "data" / "processed" / "equity_trade_log.csv"
    return list(_iter_csv(path)) if path.is_file() else []


def options_book_snapshot(root: Path) -> BookSnapshot:
    seed = large_options_book_seed()
    rows = _load_options_rows(root)
    closed_by, latest_by = _options_closed_open_maps(rows)
    realized = sum(_float_cell(r, "unrealized_pnl") for r in closed_by.values())
    open_mtm = 0.0
    for pid, row in latest_by.items():
        if pid in closed_by:
            continue
        open_mtm += _float_cell(row, "unrealized_pnl")
    return BookSnapshot(seed=seed, closed_realized=realized, open_mtm=open_mtm)


def equity_book_snapshot(root: Path) -> BookSnapshot:
    seed = large_equity_book_seed()
    rows = _load_equity_rows(root)
    closed_by, latest_by = _equity_closed_open_maps(rows)
    realized = sum(_float_cell(r, "unrealized_pnl") for r in closed_by.values())
    open_mtm = 0.0
    for pid, row in latest_by.items():
        if pid in closed_by:
            continue
        open_mtm += _float_cell(row, "unrealized_pnl")
    return BookSnapshot(seed=seed, closed_realized=realized, open_mtm=open_mtm)


def write_trading_ledgers(root: Path) -> dict[str, Path]:
    """Rebuild all three ledger CSVs from on-disk logs + challenge state."""
    out_dir = ledger_dir(root)
    paths: dict[str, Path] = {}
    as_of = datetime.now(tz=timezone.utc).isoformat()

    # --- Large options ---
    opt_snap = options_book_snapshot(root)
    opt_rows_data = _load_options_rows(root)
    closed_by, _latest_by = _options_closed_open_maps(opt_rows_data)
    opt_rows: list[list[Any]] = [
        [
            "large_options",
            "SEED",
            "",
            "",
            as_of,
            0.0,
            0.0,
            opt_snap.seed,
            "Notional book start (advanced options sleeve)",
        ]
    ]
    closed_list = sorted(
        closed_by.items(),
        key=lambda x: (_parse_ts(x[1].get("timestamp_utc", "")) or datetime.min.replace(tzinfo=timezone.utc)),
    )
    running_closed = opt_snap.seed
    for pid, row in closed_list:
        pnl = _float_cell(row, "unrealized_pnl")
        running_closed += pnl
        opt_rows.append(
            [
                "large_options",
                "CLOSED",
                pid,
                str(row.get("symbol", "")),
                row.get("timestamp_utc", ""),
                pnl,
                running_closed,
                running_closed + opt_snap.open_mtm,
                str(row.get("note", ""))[:120],
            ]
        )
    opt_rows.append(
        [
            "large_options",
            "OPEN_MTM_SUM",
            "",
            "",
            as_of,
            opt_snap.open_mtm,
            running_closed,
            opt_snap.book_value,
            "Mark-to-model unrealized on open plan_ids (monitor CSV)",
        ]
    )
    opt_path = out_dir / "large_options_book.csv"
    _write_ledger(opt_path, opt_rows)
    paths["large_options"] = opt_path

    # --- Large equities ---
    eq_snap = equity_book_snapshot(root)
    eq_data = _load_equity_rows(root)
    eq_closed_by, eq_latest_by = _equity_closed_open_maps(eq_data)
    eq_rows: list[list[Any]] = [
        [
            "large_equities",
            "SEED",
            "",
            "",
            as_of,
            0.0,
            0.0,
            eq_snap.seed,
            "Notional book start (prop-style IBKR equity sleeve)",
        ]
    ]
    eq_closed_list = sorted(
        eq_closed_by.items(),
        key=lambda x: (_parse_ts(x[1].get("timestamp_utc", "")) or datetime.min.replace(tzinfo=timezone.utc)),
    )
    run_eq: float = eq_snap.seed
    for pid, row in eq_closed_list:
        pnl = _float_cell(row, "unrealized_pnl")
        run_eq += pnl
        eq_rows.append(
            [
                "large_equities",
                "CLOSED",
                pid,
                str(row.get("symbol", "")),
                row.get("timestamp_utc", ""),
                pnl,
                run_eq,
                run_eq + eq_snap.open_mtm,
                str(row.get("note", ""))[:120],
            ]
        )
    eq_rows.append(
        [
            "large_equities",
            "OPEN_MTM_SUM",
            "",
            "",
            as_of,
            eq_snap.open_mtm,
            run_eq,
            eq_snap.book_value,
            "Open equity positions (last log mark)",
        ]
    )
    eq_path = out_dir / "large_equities_book.csv"
    _write_ledger(eq_path, eq_rows)
    paths["large_equities"] = eq_path

    # --- PDT challenge (Robinhood narrative; still RLM dry-run state) ---
    ch_path_state = root / "data" / "challenge" / "state.json"
    pdt_rows: list[list[Any]] = []
    if ch_path_state.is_file():
        try:
            data = json.loads(ch_path_state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        bal = float(data.get("balance", 0.0))
        seed = float(data.get("seed", 1000.0))
        pdt_rows.append(
            [
                "pdt_robinhood",
                "SEED",
                "",
                "",
                str(data.get("created_at", as_of)),
                0.0,
                seed,
                seed,
                "Robinhood-style $1K→$25K challenge (RLM paper state)",
            ]
        )
        th_raw = data.get("trade_history") or []
        th_list = [t for t in th_raw if isinstance(t, dict)] if isinstance(th_raw, list) else []

        def _exit_key(t: dict) -> str:
            return str(t.get("exit_date", ""))[:19]

        th_list.sort(key=_exit_key)
        run = seed
        for t in th_list:
            try:
                pnl = float(t.get("pnl", 0.0) or 0.0)
            except (TypeError, ValueError):
                pnl = 0.0
            run += pnl
            pdt_rows.append(
                [
                    "pdt_robinhood",
                    "ROUND_TRIP",
                    str(t.get("trade_id", "")),
                    str(t.get("symbol", "")),
                    str(t.get("exit_date", "")),
                    pnl,
                    run,
                    run,
                    str(t.get("exit_reason", ""))[:120],
                ]
            )
        o_pos = data.get("open_positions") or []
        open_mtm_c = 0.0
        if isinstance(o_pos, list):
            for p in o_pos:
                if isinstance(p, dict) and str(p.get("status", "open")) == "open":
                    try:
                        open_mtm_c += float(p.get("unrealised_pnl", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        pass
        pdt_rows.append(
            [
                "pdt_robinhood",
                "OPEN_MTM_SUM",
                "",
                "",
                as_of,
                open_mtm_c,
                bal,
                bal + open_mtm_c,
                "State balance + open-leg MTM (challenge dry-run)",
            ]
        )
    else:
        pdt_rows.append(
            [
                "pdt_robinhood",
                "UNINITIALIZED",
                "",
                "",
                as_of,
                0.0,
                0.0,
                0.0,
                "Run: rlm challenge --reset",
            ]
        )
    pdt_path = out_dir / "pdt_robinhood_challenge_book.csv"
    _write_ledger(pdt_path, pdt_rows)
    paths["pdt_robinhood"] = pdt_path

    return paths
