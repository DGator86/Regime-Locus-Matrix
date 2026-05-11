"""Reset on-disk **paper** equities + options monitor state (local JSON/CSV only).

Does **not** flatten IBKR paper positions — close those in TWS / Mobile if needed.

Clears / archives under ``<repo>/data/processed``:

- ``equity_positions_state.json``, ``equity_trade_log.csv``
- options monitor CSVs (``trade_log.csv``, env ``RLM_OPTIONS_TRADE_LOG_PATH``, and
  ``options_large_account_trade_log.csv`` when distinct)
- ``trade_monitor_state.json`` (monitor peaks / trail / paper_close_sent)
- ``telegram_notify_state.json`` (option alerts de-dupe)

Optional: ``--with-challenge`` runs the same reset as ``rlm challenge --reset``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.tracker import ChallengeTracker
from rlm.cli.common import add_data_root_arg
from rlm.data.paths import get_repo_root
from rlm.notify.options_paths import options_trade_log_primary, options_trade_log_read_paths

# Keep in sync with scripts/monitor_active_trade_plans.py::_TRADE_LOG_COLUMNS
_OPTIONS_LOG_COLUMNS = [
    "timestamp_utc",
    "plan_id",
    "symbol",
    "strategy",
    "entry_debit",
    "entry_mid",
    "current_mark",
    "peak_mark",
    "unrealized_pnl",
    "unrealized_pnl_pct",
    "signal",
    "closed",
    "dte",
]

# Keep in sync with scripts/ibkr_equity_paper_trade.py::_LOG_COLUMNS
_EQUITY_LOG_COLUMNS = [
    "timestamp_utc",
    "plan_id",
    "symbol",
    "strategy",
    "action",
    "quantity",
    "current_mark",
    "entry_debit",
    "order_id",
    "unrealized_pnl",
    "unrealized_pnl_pct",
    "signal",
    "closed",
    "note",
]

_TELEGRAM_NOTIFY_EMPTY: dict[str, object] = {
    "notify_seeded": False,
    "announced_trade_open": [],
    "last_opt_signal": {},
    "announced_tp": [],
    "announced_exit": [],
    "last_equity_open": [],
    "announced_equity_close": [],
    "last_universe_active_ids": [],
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--no-archive",
        action="store_true",
        help="Delete prior files instead of renaming to *.reset_<UTC>.bak",
    )
    p.add_argument(
        "--with-challenge",
        action="store_true",
        help="Also reset data/challenge (same as rlm challenge --reset)",
    )
    p.add_argument(
        "--challenge-capital",
        type=float,
        default=1_000.0,
        metavar="USD",
        help="Seed balance for --with-challenge (default: 1000)",
    )
    p.add_argument(
        "--challenge-target",
        type=float,
        default=25_000.0,
        metavar="USD",
        help="Target for --with-challenge (default: 25000)",
    )
    add_data_root_arg(p)
    return p.parse_args()


def _data_parent(args: argparse.Namespace) -> Path:
    """Directory containing ``processed/`` (default: ``<repo>/data``)."""
    if args.data_root:
        return Path(args.data_root).expanduser().resolve()
    return (get_repo_root() / "data").resolve()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _archive_or_drop(path: Path, *, stamp: str, archive: bool) -> None:
    if not path.is_file():
        return
    if path.stat().st_size == 0:
        path.unlink()
        return
    if archive:
        dest = path.with_name(f"{path.name}.reset_{stamp}.bak")
        if dest.exists():
            dest = path.with_name(f"{path.name}.reset_{stamp}.bak.{id(path)}")
        path.replace(dest)
    else:
        path.unlink()


def _write_json(path: Path, obj: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _write_csv_header_only(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()


def reset_paper_books(
    *,
    data_parent: Path,
    archive: bool,
    stamp: str,
) -> list[str]:
    """Perform reset. Returns human-readable log lines."""
    repo = get_repo_root()
    proc = data_parent / "processed"
    lines: list[str] = []

    # --- JSON state ---
    for name, payload in (
        ("equity_positions_state.json", {}),
        ("trade_monitor_state.json", {}),
        ("telegram_notify_state.json", _TELEGRAM_NOTIFY_EMPTY),
    ):
        path = proc / name
        if path.is_file():
            _archive_or_drop(path, stamp=stamp, archive=archive)
            lines.append(f"archived/dropped: {path.relative_to(data_parent.parent)}")
        _write_json(path, payload)  # type: ignore[arg-type]
        lines.append(f"written fresh: {path.relative_to(data_parent.parent)}")

    # --- Equity trade log ---
    eq_log = proc / "equity_trade_log.csv"
    if eq_log.is_file():
        _archive_or_drop(eq_log, stamp=stamp, archive=archive)
    _write_csv_header_only(eq_log, _EQUITY_LOG_COLUMNS)
    lines.append(f"header-only: {eq_log.relative_to(data_parent.parent)}")

    # --- Options monitor logs (all resolved paths, de-duped) ---
    opt_paths: list[Path] = []
    seen: set[str] = set()
    for cand in options_trade_log_read_paths(repo):
        key = str(cand.resolve())
        if key in seen:
            continue
        seen.add(key)
        opt_paths.append(cand)
    primary = options_trade_log_primary(repo)
    pk = str(primary.resolve())
    if pk not in seen:
        opt_paths.append(primary)

    for opt_log in opt_paths:
        try:
            rel = opt_log.relative_to(repo)
        except ValueError:
            rel = opt_log
        if opt_log.is_file():
            _archive_or_drop(opt_log, stamp=stamp, archive=archive)
        _write_csv_header_only(opt_log, _OPTIONS_LOG_COLUMNS)
        lines.append(f"header-only: {rel}")

    return lines


def main() -> None:
    args = _parse_args()
    stamp = _stamp()
    data_parent = _data_parent(args)
    if not data_parent.is_dir():
        print(f"Data directory not found: {data_parent}", file=sys.stderr)
        sys.exit(1)

    print("Resetting paper equities + options monitor state...")
    print(f"  data root: {data_parent}")
    print(f"  archive prior: {not args.no_archive}  (stamp {stamp})")
    print("  NOTE: IBKR paper positions are NOT closed -- flatten in TWS if you want a flat book.")
    for line in reset_paper_books(data_parent=data_parent, archive=not args.no_archive, stamp=stamp):
        print(f"  {line}")

    if args.with_challenge:
        sym = "SPY"
        cfg = ChallengeConfig(
            seed_capital=args.challenge_capital,
            target_capital=args.challenge_target,
            symbol=sym,
        )
        # ChallengeTracker interprets data_root as parent of challenge/
        tracker = ChallengeTracker(data_root=str(data_parent))
        state = tracker.reset(cfg)
        print(
            f"Challenge reset: balance=${state.balance:,.2f} target=${state.target:,.2f} "
            f"({tracker.state_path()})"
        )


if __name__ == "__main__":
    main()
