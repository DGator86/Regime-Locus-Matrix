"""Regime-directed equity paper trading via IBKR.

Reads ``universe_trade_plans.json`` (output of ``run_universe_options_pipeline.py``),
extracts the regime direction for every active plan (``regime_direction``, else the first
segment of ``regime_key``, else inference from options ``strategy_name`` such as
``debit_spread_call`` / ``debit_spread_put``), and places simple stock
BUY / SELL orders on the IBKR paper account.

This runs alongside the options book for independent execution verification:
- No options permissions required — plain equity orders work at any account level.
- Bull regime  → BUY shares
- Bear regime  → SELL (short) shares
- Range / other → skip

**Prop-firm daily gates (optional):** ``RLM_EQUITY_PROP_MAX_DAILY_LOSS_PCT`` (default ``2.0``)
times ``RLM_LARGE_EQUITY_BOOK_SEED`` (default ``250000``) defines a **floor** on today's
realized equity P&L (US/Eastern session, from ``equity_trade_log.csv``); new entries
halt if the floor is touched. ``RLM_EQUITY_MAX_NEW_OPENS_PER_DAY`` (default ``4``)
limits fresh tickets per NY day. Set a knob to ``0`` to disable that side.

Positions are tracked in ``equity_positions_state.json``.  On each run the
script evaluates open equity positions against the **fresh** universe row for
each ``plan_id`` (when still present): ROEE regime flip
(:func:`~rlm.roee.exits.should_exit_for_regime_flip`), optional min top-1
**transition-matrix** next-step probability (``RLM_EQUITY_MIN_MOST_LIKELY_NEXT_PROB``),
optional **label-aligned** mass (``RLM_EQUITY_MIN_NEXT_LABEL_ALIGNED_MASS``),
hard stop / take-profit, optional **trailing give-back** from best price since
arm (discretionary style: let winners run, lock in after a favorable move),
and optional plan-catalog absence exit (off by default — pros do not flatten
stocks just because a JSON row rotated).

**Dynamic horizon** (default on via ``RLM_EQUITY_DYNAMIC_HORIZON``): at entry,
stop / target / min-hold / trail parameters are blended from pipeline context
(primary bar size, multi-timeframe confirmation, transition stability, size
fraction) so a 5‑minute-aligned entry can wear a short leash while a slower,
better-confirmed thesis can behave like a multi-day swing without named “modes.”

Thesis-driven exits (regime flip, transition gates) honour **minimum hold**
(stored per position when dynamic horizon is active) so a single noisy refresh
does not flip the book; hard stops always apply immediately.

Usage
-----
    python scripts/ibkr_equity_paper_trade.py
    python scripts/ibkr_equity_paper_trade.py --dry-run          # no real orders
    python scripts/ibkr_equity_paper_trade.py --position-usd 5000
    python scripts/ibkr_equity_paper_trade.py --stop-pct 3 --target-pct 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Type

_BAR_SIZE_MINUTES_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*"
    r"(sec|secs|second|seconds|min|mins|minute|minutes|hr|hrs|hour|hours|"
    r"day|days|wk|week|weeks)s?\s*$",
    re.IGNORECASE,
)

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("RLM_ROOT", str(REPO_ROOT))).expanduser().resolve()
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Optional ibapi dependency — required for live IBKR connectivity, not needed for --dry-run.
try:
    from ibapi.client import EClient as _EClient
    from ibapi.wrapper import EWrapper as _EWrapper
    from ibapi.contract import Contract as _IbkrContract
    from ibapi.order import Order as _IbkrOrder
    _IBAPI_OK = True
except ImportError:
    _EClient = _EWrapper = _IbkrContract = _IbkrOrder = None  # type: ignore[assignment,misc]
    _IBAPI_OK = False

from rlm.utils.compute_threads import apply_compute_thread_env  # noqa: E402
apply_compute_thread_env()

from rlm.data.ibkr_snapshot import fetch_ibkr_account_snapshot
from rlm.regimes.forecast_regime_snapshot import (
    plan_regime_key,
    position_directional_transition_mass,
    regime_direction_equity,
    regime_transition_best_prob,
)
from rlm.roee.exits import should_exit_for_regime_flip
from rlm.universe.active_plans import iter_active_trade_plan_rows  # noqa: E402

PLANS_PATH = DATA_ROOT / "data" / "processed" / "universe_trade_plans.json"
EQUITY_STATE_PATH = DATA_ROOT / "data" / "processed" / "equity_positions_state.json"
EQUITY_LOG_PATH = DATA_ROOT / "data" / "processed" / "equity_trade_log.csv"

IBKR_LIVE_PORTS: frozenset[int] = frozenset({7496, 4001})
IBKR_PAPER_PORTS: frozenset[int] = frozenset({7497, 4002, 4004})

# IBKR error codes that are advisory notices, not hard order rejections.
# These are silently swallowed by the error handler so they never enter
# _error_lines and never trigger false-positive rejection raises.
#   2104-2174  market-data / connectivity info
#   10349      "Order TIF was set to DAY based on order preset"
#   10314      "Order modified to comply with …"
#   10197      "No market data during competing session"
_IBKR_ADVISORY_CODES: frozenset[int] = frozenset({
    2104, 2106, 2107, 2108, 2158, 2174,
    10197, 10314, 10349,
})

_LOG_COLUMNS = [
    "timestamp_utc", "plan_id", "symbol", "strategy", "action",
    "quantity", "current_mark", "entry_debit", "order_id",
    "unrealized_pnl", "unrealized_pnl_pct", "signal", "closed", "note",
]


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class EquityPosition:
    plan_id: str
    symbol: str
    direction: str          # "bull" | "bear"
    side: str               # "long" | "short"
    quantity: int
    entry_price: float
    entry_ts: str           # ISO UTC
    ibkr_order_id: int | None = None
    close_order_id: int | None = None
    status: str = "open"    # "open" | "closed"
    exit_price: float | None = None
    exit_ts: str | None = None
    exit_reason: str | None = None
    plan_missing_since_utc: str | None = None
    entry_regime_key: str = ""
    # Trailing give-back from best price while armed (see evaluate_equity_positions).
    trail_armed: bool = False
    trail_peak_price: float | None = None
    # Set at entry when dynamic horizon is on; None => use evaluate() globals (legacy state).
    eff_stop_pct: float | None = None
    eff_target_pct: float | None = None
    eff_min_hold_sec: float | None = None
    eff_trail_activate_pct: float | None = None
    eff_trail_retrace_frac: float | None = None
    eff_profile_note: str = ""


def _load_state(path: Path) -> dict[str, EquityPosition]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, EquityPosition] = {}
    for pid, d in raw.items():
        try:
            out[pid] = EquityPosition(**d)
        except TypeError:
            pass
    return out


def _save_state(positions: dict[str, EquityPosition], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {pid: asdict(pos) for pid, pos in positions.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_iso_utc(ts: str) -> datetime:
    s = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _equity_prop_firm_new_entry_ok(log_path: Path) -> tuple[bool, str]:
    """Prop-style gates: stop adding risk after a red NY day or too many fresh tickets.

    Disable loss cap with ``RLM_EQUITY_PROP_MAX_DAILY_LOSS_PCT=0``; disable open cap
    with ``RLM_EQUITY_MAX_NEW_OPENS_PER_DAY=0``.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    max_loss_spec = (os.environ.get("RLM_EQUITY_PROP_MAX_DAILY_LOSS_PCT") or "2.0").strip()
    max_opens_spec = (os.environ.get("RLM_EQUITY_MAX_NEW_OPENS_PER_DAY") or "4").strip()
    try:
        max_loss_pct = float(max_loss_spec or "2.0")
    except ValueError:
        max_loss_pct = 2.0
    try:
        max_opens = int(float(max_opens_spec or "4"))
    except ValueError:
        max_opens = 4
    if max_loss_pct <= 0 and max_opens <= 0:
        return True, ""
    try:
        book_seed = float((os.environ.get("RLM_LARGE_EQUITY_BOOK_SEED") or "250000").strip())
    except ValueError:
        book_seed = 250_000.0
    today_et = datetime.now(tz=et).date()
    realized_today = 0.0
    opens_today = 0
    if log_path.is_file():
        try:
            with log_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        ts = _parse_iso_utc(str(row.get("timestamp_utc", "")))
                    except ValueError:
                        continue
                    if ts.astimezone(et).date() != today_et:
                        continue
                    if (row.get("closed") or "0").strip() == "1":
                        try:
                            realized_today += float(row.get("unrealized_pnl") or 0.0)
                        except (TypeError, ValueError):
                            pass
                    if str(row.get("signal", "")).strip().lower() == "open":
                        opens_today += 1
        except OSError:
            pass
    if max_loss_pct > 0 and book_seed > 0:
        floor = -book_seed * (max_loss_pct / 100.0)
        if realized_today <= floor - 1e-9:
            return (
                False,
                f"daily realized {realized_today:+.2f} hit floor {floor:+.2f} "
                f"({max_loss_pct:.2f}% of ${book_seed:,.0f} book seed)",
            )
    if max_opens > 0 and opens_today >= max_opens:
        return False, f"opens today {opens_today} >= cap {max_opens}"
    return True, ""


def _plan_missing_grace_sec(cli_value: float | None) -> float:
    if cli_value is not None:
        return max(0.0, float(cli_value))
    raw = (os.environ.get("RLM_EQUITY_PLAN_MISSING_GRACE_SEC") or "").strip()
    if raw:
        return max(0.0, float(raw))
    return 900.0


def _exit_on_plan_absent(cli_flag: bool | None) -> bool:
    """Exit when plan_id leaves the active universe after grace. Default on."""
    if cli_flag is True:
        return True
    if cli_flag is False:
        return False
    raw = (os.environ.get("RLM_EQUITY_EXIT_ON_PLAN_ABSENT") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _min_hold_thesis_sec(cli_value: float | None) -> float:
    """Seconds after entry before regime / transition-model exits may fire (stops/trails/TP always apply)."""
    if cli_value is not None:
        return max(0.0, float(cli_value))
    raw = (os.environ.get("RLM_EQUITY_MIN_HOLD_THESIS_SEC") or "").strip()
    if raw:
        return max(0.0, float(raw))
    return 1800.0


def _trail_activate_pct(cli_value: float | None) -> float | None:
    """Arm trailing give-back once unrealized PnL%% reaches this; None disables."""
    if cli_value is not None:
        v = float(cli_value)
        return None if v <= 0.0 else v
    raw = (os.environ.get("RLM_EQUITY_TRAIL_ACTIVATE_PCT") or "").strip()
    if raw:
        v = float(raw)
        return None if v <= 0.0 else v
    if (os.environ.get("RLM_EQUITY_TRAIL_DISABLE") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    return 8.0


def _trail_retrace_frac(cli_value: float | None) -> float | None:
    """Fraction of MFE (move from entry to peak favorable price) to give back before trail exit."""
    if cli_value is not None:
        v = float(cli_value)
        return None if v <= 0.0 else min(1.0, v)
    raw = (os.environ.get("RLM_EQUITY_TRAIL_RETRACE_FRAC") or "").strip()
    if raw:
        v = float(raw)
        return None if v <= 0.0 else min(1.0, v)
    return 0.25


def _min_trail_exit_pnl_pct(cli_value: float | None) -> float:
    """Do not trail-exit unless unrealized PnL%% is still at/above this floor (default 0 = breakeven)."""
    if cli_value is not None:
        return float(cli_value)
    raw = (os.environ.get("RLM_EQUITY_MIN_TRAIL_EXIT_PCT") or "").strip()
    if raw:
        return float(raw)
    return 0.0


def _dynamic_horizon_enabled(cli_flag: bool | None) -> bool:
    """Blend stop/target/min-hold/trail from plan context (bar size, TF confirm, transition, size)."""
    if cli_flag is True:
        return True
    if cli_flag is False:
        return False
    raw = (os.environ.get("RLM_EQUITY_DYNAMIC_HORIZON") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return True


def _env_float(name: str, default: str) -> float:
    s = (os.environ.get(name) or "").strip()
    return float(default) if not s else float(s)


def _bar_size_to_minutes(bar_size: str) -> float | None:
    s = str(bar_size or "").strip()
    if not s:
        return None
    m = _BAR_SIZE_MINUTES_RE.match(s)
    if not m:
        return None
    val = float(m.group(1))
    u = str(m.group(2)).lower()
    if u.startswith("sec"):
        return val / 60.0
    if u.startswith("min"):
        return val
    if u.startswith("hr") or u.startswith("hour"):
        return val * 60.0
    if u.startswith("day"):
        return val * 1440.0
    if u.startswith("wk") or u.startswith("week"):
        return val * 10080.0
    return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _horizon_blend_from_plan(plan: dict[str, Any]) -> float:
    """0 ≈ short-cycle (tight leash), 1 ≈ slower / swing-style book-keeping."""
    pipe: dict[str, Any] = {}
    p_raw = plan.get("pipeline")
    if isinstance(p_raw, dict):
        pipe = p_raw
    bar_s = str(pipe.get("bar_size") or plan.get("primary_bar_size") or "").strip()
    bmin = _bar_size_to_minutes(bar_s)
    if bmin is None or bmin <= 0:
        bar_part = 0.5
    else:
        lo = math.log(6.0)
        hi = math.log(8000.0)
        bar_part = _clamp01((math.log(bmin + 1.0) - lo) / max(1e-9, hi - lo))
    tf_fail = pipe.get("tf_confirmation_failed")
    if tf_fail is False:
        tf_part = 1.0
    elif tf_fail is True:
        tf_part = 0.55
    else:
        tf_part = 0.72
    trans = pipe.get("regime_transition")
    p_next = regime_transition_best_prob(trans if isinstance(trans, dict) else None)
    stab = float(p_next) if p_next is not None else 0.55
    stab = _clamp01(stab)
    dec = plan.get("decision")
    conf = 0.75
    if isinstance(dec, dict):
        try:
            conf = float(dec.get("size_fraction") or conf)
        except (TypeError, ValueError):
            conf = 0.75
    conf = _clamp01(conf)
    h = 0.38 * bar_part + 0.28 * tf_part + 0.22 * stab + 0.12 * conf
    return _clamp01(h)


def _resolve_equity_trade_params(
    plan: dict[str, Any],
    *,
    dynamic_horizon: bool,
    stop_pct: float,
    target_pct: float,
    min_hold_sec: float,
    trail_activate_pct: float | None,
    trail_retrace_frac: float | None,
) -> tuple[float, float, float, float | None, float | None, str]:
    """Return effective (stop_pct, target_pct, min_hold_sec, trail_activate, trail_retrace, note)."""
    if not dynamic_horizon:
        return (
            float(stop_pct),
            float(target_pct),
            float(min_hold_sec),
            trail_activate_pct,
            trail_retrace_frac,
            "static",
        )
    h = _horizon_blend_from_plan(plan)
    hold_lo = max(60.0, _env_float("RLM_EQUITY_DYN_MIN_HOLD_FLOOR_SEC", "300"))
    hold_hi = max(hold_lo, _env_float("RLM_EQUITY_DYN_MIN_HOLD_CAP_SEC", "172800"))
    stop_lo = max(0.1, _env_float("RLM_EQUITY_DYN_STOP_PCT_MIN", "0.9"))
    stop_hi = max(stop_lo, _env_float("RLM_EQUITY_DYN_STOP_PCT_MAX", "6.5"))
    tgt_lo = max(0.1, _env_float("RLM_EQUITY_DYN_TARGET_PCT_MIN", "1.0"))
    tgt_hi = max(tgt_lo, _env_float("RLM_EQUITY_DYN_TARGET_PCT_MAX", "18.0"))

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    mh = lerp(hold_lo, hold_hi, h)
    sp = lerp(stop_lo, stop_hi, h)
    tgp = lerp(tgt_lo, tgt_hi, h)
    if trail_activate_pct is None or trail_retrace_frac is None:
        tact, trf = None, None
    else:
        ta_lo = max(0.05, _env_float("RLM_EQUITY_DYN_TRAIL_ACT_MIN_PCT", "0.45"))
        ta_hi = max(ta_lo, _env_float("RLM_EQUITY_DYN_TRAIL_ACT_MAX_PCT", "5.0"))
        tr_lo = max(0.05, _env_float("RLM_EQUITY_DYN_TRAIL_RETRACE_MIN", "0.22"))
        tr_hi = max(tr_lo, _env_float("RLM_EQUITY_DYN_TRAIL_RETRACE_MAX", "0.48"))
        tact = lerp(ta_lo, ta_hi, h)
        trf = min(1.0, max(0.05, lerp(tr_lo, tr_hi, h)))
    return sp, tgp, mh, tact, trf, f"dyn_h={h:.2f}"


def _position_age_sec(entry_ts: str, now: datetime) -> float:
    try:
        return max(0.0, (now - _parse_iso_utc(entry_ts)).total_seconds())
    except (TypeError, ValueError, OSError):
        return float("inf")


def _plans_by_plan_id(plans: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in plans:
        pid = str(p.get("plan_id") or "")
        if pid:
            out[pid] = p
    return out


def _merge_universe_rows_for_open_positions(
    plan_by_id: dict[str, dict],
    plans_path: Path,
    positions: dict[str, EquityPosition],
) -> None:
    """Fill plan snapshots for open legs whose plan_id is not in the active set (e.g. trimmed row)."""
    missing = {
        pid
        for pid, pos in positions.items()
        if pos.status == "open" and pid not in plan_by_id
    }
    if not missing or not plans_path.is_file():
        return
    try:
        raw = json.loads(plans_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for row in (raw.get("active_ranked") or []) + (raw.get("results") or []):
        if not isinstance(row, dict):
            continue
        pid = str(row.get("plan_id") or "")
        if pid in missing:
            plan_by_id[pid] = row


def _min_most_likely_next_prob(cli_value: float | None) -> float | None:
    """Optional threshold on calibrated/top-1 next-step prob (unset = disable)."""
    if cli_value is not None:
        v = float(cli_value)
        return v if v > 0.0 else None
    raw = (os.environ.get("RLM_EQUITY_MIN_MOST_LIKELY_NEXT_PROB") or "").strip()
    if not raw:
        return None
    v = float(raw)
    return v if v > 0.0 else None


def _min_next_label_aligned_mass(cli_value: float | None) -> float | None:
    """Optional min mass on Σ P(next_state)×label_alignment for traded direction."""
    if cli_value is not None:
        v = float(cli_value)
        return v if v > 0.0 else None
    raw = (os.environ.get("RLM_EQUITY_MIN_NEXT_LABEL_ALIGNED_MASS") or "").strip()
    if not raw:
        return None
    v = float(raw)
    return v if v > 0.0 else None


# ---------------------------------------------------------------------------
# CSV trade log
# ---------------------------------------------------------------------------

def _append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_LOG_COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def _ensure_equity_log_file(path: Path) -> None:
    """Create CSV with header so EOD / dashboards see the book even before first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_LOG_COLUMNS, extrasaction="ignore")
        w.writeheader()


# ---------------------------------------------------------------------------
# IBKR connectivity (stock orders)
# ---------------------------------------------------------------------------

def _load_equity_socket_config() -> tuple[str, int, int]:
    host = (os.environ.get("IBKR_HOST") or "127.0.0.1").strip()
    port = int(os.environ.get("IBKR_PORT") or "7497")
    cid = int(os.environ.get("IBKR_EQUITY_CLIENT_ID") or "10")
    return host, port, cid


def _ibkr_order_outcome(status: str, *, filled: float = 0.0, remaining: float = 0.0) -> str:
    """Classify an IBKR orderStatus callback as filled / failed / pending.

    Only ``filled`` means local equity state may record an open or close.
    ``PreSubmitted`` / ``Submitted`` are pending (not success). ``Cancelled`` /
    ``ApiCancelled`` / ``Rejected`` are hard failures.
    """
    last = (status or "").strip()
    if last in {"Cancelled", "ApiCancelled", "Rejected", "Inactive"}:
        return "failed"
    if last == "Filled" or (float(filled) > 0.0 and float(remaining) <= 1e-9):
        return "filled"
    return "pending"


def _ibkr_order_update(
    status: str,
    *,
    filled: float = 0.0,
    remaining: float = 0.0,
    avg_fill_price: float = 0.0,
) -> dict[str, Any]:
    """Normalize one orderStatus callback into a fill-aware status record."""
    return {
        "status": (status or "").strip(),
        "filled": float(filled or 0.0),
        "remaining": float(remaining or 0.0),
        "avg_fill_price": float(avg_fill_price or 0.0),
    }


def _ibkr_fill_meta_from_updates(updates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return fill metadata when the latest update is a completed fill; else None."""
    if not updates:
        return None
    last = updates[-1]
    outcome = _ibkr_order_outcome(
        str(last.get("status") or ""),
        filled=float(last.get("filled") or 0.0),
        remaining=float(last.get("remaining") or 0.0),
    )
    if outcome != "filled":
        return None
    return {
        "status": str(last.get("status") or ""),
        "filled": float(last.get("filled") or 0.0),
        "remaining": float(last.get("remaining") or 0.0),
        "avg_fill_price": float(last.get("avg_fill_price") or 0.0),
        "trail": [str(u.get("status") or "") for u in updates],
    }


def _ibkr_order_failed_message(
    order_id: int,
    updates: list[dict[str, Any]],
    hard_errs: list[tuple[int, str]],
) -> str | None:
    """Human-readable failure for a hard reject/cancel, or None if still pending."""
    if hard_errs:
        return f"IBKR order {order_id} rejected: {hard_errs[-1]}"
    if not updates:
        return None
    last = updates[-1]
    outcome = _ibkr_order_outcome(
        str(last.get("status") or ""),
        filled=float(last.get("filled") or 0.0),
        remaining=float(last.get("remaining") or 0.0),
    )
    if outcome != "failed":
        return None
    status = str(last.get("status") or "failed")
    return f"IBKR order {order_id} {status}: {status}"


def _coerce_ibkr_fill_meta(fill_meta: Any) -> dict[str, Any]:
    """Accept only mapping fill metadata; reject legacy list[str] trails as unfilled."""
    if isinstance(fill_meta, dict):
        return fill_meta
    return {}


def _get_ibapi_bundle() -> tuple[Type[Any], Any]:
    """Return (EClient, EWrapper); raise SystemExit if ibapi is not installed."""
    if not _IBAPI_OK:
        raise SystemExit("ibapi not installed. Run: pip install ibapi")
    return _EClient, _EWrapper


class _EquityApp:
    """Minimal IBKR EWrapper/EClient combo for stock orders."""

    def __init__(self) -> None:
        EClient, EWrapper = _get_ibapi_bundle()

        class _App(EWrapper, EClient):  # type: ignore[misc]
            def __init__(inner_self) -> None:
                EWrapper.__init__(inner_self)
                EClient.__init__(inner_self, inner_self)

        self._app = _App()
        # Latest orderStatus callbacks per order id (status + fill qty/price).
        self._app._order_status: dict[int, list[dict[str, Any]]] = {}
        self._app._error_lines: list[tuple[int, int, str]] = []
        self._app._next_order_id: int | None = None
        self._app._ticker_prices: dict[int, float] = {}
        self._app._ticker_events: dict[int, threading.Event] = {}
        # Separate counter for market-data reqIds so they never collide with order IDs.
        # Order IDs start from whatever IBKR assigns (typically small integers) and
        # count upward.  Market-data reqIds start at 10_000 and count downward,
        # keeping them well away from the order-ID sequence in both directions.
        self._mkt_req_counter: int = 10_000

        original_error = self._app.error.__func__ if hasattr(self._app.error, "__func__") else None

        def _error(reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
            if errorCode in _IBKR_ADVISORY_CODES:
                return  # purely informational — do not record or print
            print(f"  [ibkr-err] reqId={reqId} code={errorCode} {errorString}", flush=True)
            if reqId != -1:
                self._app._error_lines.append((reqId, errorCode, errorString))

        def _order_status(orderId: int, status: str, filled: float, remaining: float,
                          avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                          clientId: int, whyHeld: str, mktCapPrice: float = 0.0) -> None:
            self._app._order_status.setdefault(orderId, []).append(
                _ibkr_order_update(
                    status,
                    filled=filled,
                    remaining=remaining,
                    avg_fill_price=avgFillPrice,
                )
            )

        def _next_valid_id(orderId: int) -> None:
            self._app._next_order_id = orderId

        def _tick_price(reqId: int, tickType: int, price: float, attrib: Any) -> None:
            if tickType in (1, 2, 4, 68):  # bid, ask, last, midpoint
                self._app._ticker_prices[reqId] = price
                if reqId in self._app._ticker_events:
                    self._app._ticker_events[reqId].set()

        self._app.error = _error  # type: ignore[method-assign]
        self._app.orderStatus = _order_status  # type: ignore[method-assign]
        self._app.nextValidId = _next_valid_id  # type: ignore[method-assign]
        self._app.tickPrice = _tick_price  # type: ignore[method-assign]

    @property
    def app(self) -> Any:
        return self._app

    def connect(self, host: str, port: int, client_id: int) -> None:
        self._app.connect(host, port, clientId=client_id)
        t = threading.Thread(target=self._app.run, daemon=True)
        t.start()
        deadline = time.monotonic() + 15.0
        while self._app._next_order_id is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._app._next_order_id is None:
            raise RuntimeError("IBKR handshake timed out — is TWS/Gateway running?")
        print(f"  [equity-ibkr] connected client_id={self._app.clientId}, "
              f"next_order_id={self._app._next_order_id}", flush=True)

    def disconnect(self) -> None:
        try:
            self._app.disconnect()
        except Exception:
            pass

    def next_order_id(self) -> int:
        oid = self._app._next_order_id
        if oid is None:
            raise RuntimeError("Not connected to IBKR")
        self._app._next_order_id = oid + 1
        return oid

    def _next_mkt_req_id(self) -> int:
        """Return a market-data reqId that is separate from the order-ID sequence.

        Counts down from 10_000 so market-data reqIds move away from order IDs
        (which start low and increment upward from IBKR's nextValidId).
        """
        req_id = self._mkt_req_counter
        self._mkt_req_counter -= 1
        return req_id

    def get_last_price(self, symbol: str, timeout_sec: float = 8.0) -> float | None:
        """Request live last price via reqMktData tick snapshot.

        Uses a dedicated reqId counter (10_000+) so market-data requests never
        advance the order-ID sequence.  Returns None gracefully if the account
        lacks a real-time data subscription (IBKR error 10089).
        """
        if not _IBAPI_OK or _IbkrContract is None:
            return None
        req_id = self._next_mkt_req_id()
        contract = _IbkrContract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        ev = threading.Event()
        self._app._ticker_events[req_id] = ev

        # Clear any stale error for this reqId before subscribing.
        self._app._error_lines = [
            (r, c, m) for r, c, m in self._app._error_lines if r != req_id
        ]

        self._app.reqMktData(req_id, contract, "", True, False, [])

        # Poll in short increments so we can bail out early on error 10089
        # (no market-data subscription) rather than blocking the full timeout.
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if ev.wait(timeout=0.25):
                break
            # Check for a subscription-denial error on this reqId
            if any(r == req_id and c in (10089, 354) for r, c, _ in self._app._error_lines):
                break

        try:
            self._app.cancelMktData(req_id)
        except Exception:
            pass

        price = self._app._ticker_prices.get(req_id)
        return price

    def place_stock_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float | None = None,
        transmit: bool = True,
    ) -> int:
        if not _IBAPI_OK:
            raise SystemExit("ibapi not installed. Run: pip install ibapi")

        contract = _IbkrContract()  # type: ignore[misc]
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        order = _IbkrOrder()  # type: ignore[misc]
        order.action = action  # "BUY" or "SELL"
        order.totalQuantity = quantity
        if limit_price is not None:
            order.orderType = "LMT"
            order.lmtPrice = round(limit_price, 2)
        else:
            order.orderType = "MKT"
        order.transmit = transmit
        # ibapi sets eTradeOnly=True and firmQuoteOnly=True by default; IBKR
        # paper accounts reject both with error 10268. Clear them explicitly.
        order.eTradeOnly = False
        order.firmQuoteOnly = False

        oid = self.next_order_id()
        self._app.placeOrder(oid, contract, order)
        return oid

    def wait_for_order(self, order_id: int, timeout_sec: float = 30.0) -> dict[str, Any]:
        """Block until the order is filled, or raise on cancel/reject/timeout.

        Returns fill metadata ``{status, filled, remaining, avg_fill_price, trail}``.
        Callers must not mutate local equity state unless this returns successfully.
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            updates = list(self._app._order_status.get(order_id, []))
            hard_errs = [
                (c, m)
                for r, c, m in self._app._error_lines
                if r == order_id and c not in _IBKR_ADVISORY_CODES
            ]
            fail_msg = _ibkr_order_failed_message(order_id, updates, hard_errs)
            if fail_msg is not None:
                raise RuntimeError(fail_msg)
            fill_meta = _ibkr_fill_meta_from_updates(updates)
            if fill_meta is not None:
                return fill_meta
            time.sleep(0.1)
        updates = list(self._app._order_status.get(order_id, []))
        last = updates[-1] if updates else {}
        last_status = str(last.get("status") or "")
        raise RuntimeError(
            f"IBKR order {order_id} timed out waiting for fill; last={last_status!r}"
        )


@contextmanager
def ibkr_equity_connection() -> Generator[_EquityApp, None, None]:
    host, port, cid = _load_equity_socket_config()
    if port in IBKR_LIVE_PORTS:
        raise ValueError(
            f"Refusing automated equity orders on live port {port}. "
            "Set IBKR_PORT to a paper port (7497 / 4002)."
        )
    app = _EquityApp()
    app.connect(host, port, cid)
    try:
        yield app
    finally:
        app.disconnect()


# ---------------------------------------------------------------------------
# Plan reading helpers
# ---------------------------------------------------------------------------

def _strategy_label(plan: dict) -> str:
    s = str(plan.get("strategy_name") or plan.get("strategy") or "").strip()
    if s:
        return s
    dec = plan.get("decision")
    if isinstance(dec, dict):
        return str(dec.get("strategy_name") or dec.get("strategy") or "").strip()
    return ""


def _infer_regime_direction(plan: dict) -> str:
    """Derive bull/bear for the stock leg from universe_trade_plans rows.

    Pipeline rows usually expose ``regime_key`` (``bull|…``) but not ``regime_direction``;
    options strategy names (e.g. ``debit_spread_call``) also encode direction.
    """
    d = str(plan.get("regime_direction", "")).strip().lower()
    if d in ("bull", "bear"):
        return d

    rk = plan_regime_key(plan).strip().lower()
    if rk:
        head = rk.split("|", 1)[0].strip()
        if head in ("bull", "bear"):
            return head

    n = _strategy_label(plan).lower()
    if not n:
        return ""
    if "debit_spread_put" in n or "bear_put" in n or "short_bear" in n:
        return "bear"
    if "debit_spread_call" in n or "bull_call" in n or "small_bull" in n:
        return "bull"
    return ""


def _load_plans(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, dict):
        return []
    return iter_active_trade_plan_rows(raw)


def _mark_equity_opened(plan_id: str, plans_path: Path) -> None:
    """Stamp equity_opened=true on the plan in the JSON file."""
    if not plans_path.is_file():
        return
    try:
        payload = json.loads(plans_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    changed = False
    for section in ("active_ranked", "results"):
        for row in payload.get(section) or []:
            if isinstance(row, dict) and row.get("plan_id") == plan_id:
                row["equity_opened"] = True
                changed = True
    if changed:
        plans_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _quantity_for_symbol(
    price: float,
    position_usd: float,
    risk_usd: float | None = None,
    stop_pct: float | None = None,
    account_nlv: float | None = None,
    max_account_pct: float | None = None,
    confidence: float | None = None,
) -> int:
    if price <= 0:
        return 0

    # Option 1: Account-percentage scaling (Notional = NLV * MaxPct * Confidence)
    if account_nlv is not None and max_account_pct is not None:
        conf = confidence if confidence is not None else 1.0
        target_notional = account_nlv * (max_account_pct / 100.0) * conf
        qty = int(target_notional // price)
        return max(1, qty)

    # Option 2: Risk-based sizing (Risk $ = Fixed Risk * Confidence)
    if risk_usd is not None and risk_usd > 0 and stop_pct is not None and stop_pct > 0:
        # Quantity = Risk $ / (Price * Stop %)
        conf = confidence if confidence is not None else 1.0
        scaled_risk = risk_usd * conf
        qty = int(scaled_risk / (price * (stop_pct / 100.0)))
        return max(1, qty)

    # Option 3: Fixed notional
    return max(1, int(position_usd // price))


def open_equity_positions(
    *,
    plans: list[dict],
    positions: dict[str, EquityPosition],
    position_usd: float,
    risk_usd: float | None = None,
    stop_pct: float | None = None,
    target_pct: float | None = None,
    min_hold_sec: float | None = None,
    trail_activate_pct: float | None = None,
    trail_retrace_frac: float | None = None,
    dynamic_horizon: bool = False,
    account_nlv: float | None = None,
    max_account_pct: float | None = None,
    dry_run: bool,
    app: _EquityApp | None,
    plans_path: Path,
    log_path: Path,
) -> None:
    open_symbols = {pos.symbol.upper() for pos in positions.values() if pos.status == "open"}

    ok, gate_reason = _equity_prop_firm_new_entry_ok(log_path)
    if not ok:
        print(f"  [equity] prop-firm gate — {gate_reason} — skip new opens this run", flush=True)
        return

    for plan in plans:
        sym = str(plan.get("symbol", "")).upper()
        plan_id = str(plan.get("plan_id", ""))
        if not sym or not plan_id:
            continue
        if plan.get("equity_opened"):
            continue  # already entered
        if sym in open_symbols:
            print(f"  [equity] {sym}: already have open position — skip", flush=True)
            continue

        direction = _infer_regime_direction(plan)
        if direction not in ("bull", "bear"):
            strat = _strategy_label(plan)
            print(
                f"  [equity] {sym}: direction={direction!r} (regime_key={plan.get('regime_key')!r} "
                f"strategy={strat!r}) → skip (not bull/bear)",
                flush=True,
            )
            continue

        action = "BUY" if direction == "bull" else "SELL"
        side = "long" if direction == "bull" else "short"
        rk_entry = plan_regime_key(plan)

        # Determine entry price — pipeline stores it under plan["pipeline"]["close"]
        pipeline_data = plan.get("pipeline") or {}
        entry_price = float(
            pipeline_data.get("close")
            or plan.get("close")
            or plan.get("current_price")
            or 0.0
        )
        if entry_price <= 0 and app is not None:
            print(f"  [equity] {sym}: fetching live price …", flush=True)
            lp = app.get_last_price(sym)
            if lp and lp > 0:
                entry_price = lp
        if entry_price <= 0:
            print(f"  [equity] {sym}: cannot determine price — skip", flush=True)
            continue

        # Extract confidence from plan metadata
        decision_data = plan.get("decision") or {}
        meta = decision_data.get("metadata") or {}
        confidence = float(
            meta.get("regime_confidence")
            or meta.get("confidence")
            or decision_data.get("size_fraction")
            or 1.0
        )

        base_stop = float(stop_pct) if stop_pct is not None and float(stop_pct) > 0 else 5.0
        base_tgt = float(target_pct) if target_pct is not None and float(target_pct) > 0 else 10.0
        base_hold = float(min_hold_sec) if min_hold_sec is not None else _min_hold_thesis_sec(None)
        eff_s, eff_tgt, eff_mh, eff_ta, eff_trf, prof_note = _resolve_equity_trade_params(
            plan,
            dynamic_horizon=dynamic_horizon,
            stop_pct=base_stop,
            target_pct=base_tgt,
            min_hold_sec=base_hold,
            trail_activate_pct=trail_activate_pct,
            trail_retrace_frac=trail_retrace_frac,
        )

        qty = _quantity_for_symbol(
            entry_price,
            position_usd,
            risk_usd=risk_usd,
            stop_pct=eff_s,
            account_nlv=account_nlv,
            max_account_pct=max_account_pct,
            confidence=confidence,
        )
        if qty == 0:
            print(f"  [equity] {sym}: qty=0 at price={entry_price:.2f} — skip", flush=True)
            continue

        print(
            f"  [equity] {sym}: {action} {qty} shares @ ~${entry_price:.2f} "
            f"(${qty * entry_price:,.0f} notional) [dry={dry_run}]"
            + (f" | {prof_note} stop±{eff_s:.2f}% tgt+{eff_tgt:.2f}% hold>={eff_mh:.0f}s" if dynamic_horizon else ""),
            flush=True,
        )

        order_id: int | None = None
        fill_meta: dict[str, Any] | None = None
        if not dry_run and app is not None:
            try:
                # Use a slight slippage limit for shorts, market for longs
                lim = round(entry_price * (0.998 if direction == "bear" else 1.002), 2)
                order_id = app.place_stock_order(sym, action, qty, limit_price=lim)
                fill_meta = app.wait_for_order(order_id)
                print(f"    order_id={order_id} fill={fill_meta}", flush=True)
            except Exception as exc:
                print(f"    [equity] {sym}: order error — {exc}", flush=True)
                continue
            # Refuse ghost opens: wait_for_order must report a real fill dict.
            # Legacy list[str] trails are treated as unfilled (do not AttributeError).
            fill_meta = _coerce_ibkr_fill_meta(fill_meta)
            filled_qty = float(fill_meta.get("filled") or 0.0)
            if filled_qty <= 0.0:
                print(
                    f"    [equity] {sym}: order {order_id} returned without fill — skip open",
                    flush=True,
                )
                continue
            qty = max(1, int(filled_qty))
            avg_px = float(fill_meta.get("avg_fill_price") or 0.0)
            if avg_px > 0.0:
                entry_price = avg_px

        pos = EquityPosition(
            plan_id=plan_id,
            symbol=sym,
            direction=direction,
            side=side,
            quantity=qty,
            entry_price=entry_price,
            entry_ts=datetime.now(tz=timezone.utc).isoformat(),
            ibkr_order_id=order_id,
            status="open" if (dry_run or order_id is not None) else "pending",
            entry_regime_key=rk_entry,
            eff_stop_pct=eff_s if dynamic_horizon else None,
            eff_target_pct=eff_tgt if dynamic_horizon else None,
            eff_min_hold_sec=eff_mh if dynamic_horizon else None,
            eff_trail_activate_pct=eff_ta if dynamic_horizon else None,
            eff_trail_retrace_frac=eff_trf if dynamic_horizon else None,
            eff_profile_note=prof_note if dynamic_horizon else "",
        )
        positions[plan_id] = pos
        open_symbols.add(sym)

        if not dry_run and order_id is not None:
            _mark_equity_opened(plan_id, plans_path)

        _append_log(log_path, {
            "timestamp_utc": pos.entry_ts,
            "plan_id": plan_id,
            "symbol": sym,
            "strategy": direction,
            "action": action,
            "quantity": qty,
            "current_mark": entry_price,
            "entry_debit": entry_price,
            "order_id": order_id or "",
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "signal": "open",
            "closed": "0",
            "note": "dry_run" if dry_run else "filled",
        })


def evaluate_equity_positions(
    *,
    positions: dict[str, EquityPosition],
    active_plan_ids: set[str],
    plan_by_id: dict[str, dict],
    stop_pct: float,
    target_pct: float,
    grace_sec: float,
    min_most_likely_next_prob: float | None,
    min_next_label_aligned_mass: float | None,
    dry_run: bool,
    app: _EquityApp | None,
    log_path: Path,
    utc_now: datetime | None = None,
    exit_on_plan_absent: bool = False,
    min_hold_sec: float = 0.0,
    trail_activate_pct: float | None = None,
    trail_retrace_frac: float | None = None,
    min_trail_exit_pnl_pct: float = 0.0,
) -> None:
    now = utc_now if utc_now is not None else datetime.now(tz=timezone.utc)
    for plan_id, pos in list(positions.items()):
        if pos.status != "open":
            continue

        # Get current price
        current_price: float | None = None
        if app is not None:
            current_price = app.get_last_price(pos.symbol)
        if current_price is None or current_price <= 0:
            from rlm.data.alphavantage_quotes import fetch_cached_global_quote

            av_px = fetch_cached_global_quote(DATA_ROOT, pos.symbol)
            if av_px is not None and av_px > 0:
                current_price = av_px
                print(
                    f"  [equity] {pos.symbol}: mark from Alpha Vantage (${av_px:.2f})",
                    flush=True,
                )
        if current_price is None or current_price <= 0:
            current_price = pos.entry_price  # fallback — no change

        # P&L calculation
        if pos.side == "long":
            pnl = (current_price - pos.entry_price) * pos.quantity
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100.0
        else:
            pnl = (pos.entry_price - current_price) * pos.quantity
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100.0

        sp = float(pos.eff_stop_pct if pos.eff_stop_pct is not None else stop_pct)
        tp = float(pos.eff_target_pct if pos.eff_target_pct is not None else target_pct)
        mh = float(pos.eff_min_hold_sec if pos.eff_min_hold_sec is not None else min_hold_sec)
        tact = pos.eff_trail_activate_pct if pos.eff_trail_activate_pct is not None else trail_activate_pct
        trff = (
            None
            if tact is None
            else (
                pos.eff_trail_retrace_frac
                if pos.eff_trail_retrace_frac is not None
                else trail_retrace_frac
            )
        )

        if plan_id in active_plan_ids:
            pos.plan_missing_since_utc = None
        elif exit_on_plan_absent:
            if grace_sec > 0.0 and pos.plan_missing_since_utc is None:
                pos.plan_missing_since_utc = now.isoformat()
        else:
            pos.plan_missing_since_utc = None

        cur_plan_raw = plan_by_id.get(plan_id)
        cur_plan: dict | None = cur_plan_raw if isinstance(cur_plan_raw, dict) else None
        pipe: dict[str, Any] = {}
        if isinstance(cur_plan, dict):
            p_raw = cur_plan.get("pipeline")
            if isinstance(p_raw, dict):
                pipe = p_raw
        rt = pipe.get("regime_transition")
        trans_snap: dict[str, Any] | None = rt if isinstance(rt, dict) else None
        p_next = regime_transition_best_prob(trans_snap)
        cur_rk = plan_regime_key(cur_plan) if cur_plan else ""
        aligned_mass = position_directional_transition_mass(trans_snap, pos.direction)

        age_sec = _position_age_sec(pos.entry_ts, now)
        thesis_ok = age_sec >= float(mh)

        ek = str(pos.entry_regime_key or "").strip()
        regime_would_exit = bool(
            cur_plan and ek and cur_rk and should_exit_for_regime_flip(ek, cur_rk)
        )
        weak_prob_would_exit = bool(
            cur_plan
            and min_most_likely_next_prob is not None
            and p_next is not None
            and p_next < float(min_most_likely_next_prob)
        )
        weak_mass_would_exit = bool(
            cur_plan
            and min_next_label_aligned_mass is not None
            and aligned_mass is not None
            and aligned_mass < float(min_next_label_aligned_mass)
        )

        exit_reason: str | None = None
        if pnl_pct <= -sp:
            exit_reason = f"stop_loss_{sp:.4g}pct"
        elif (
            tact is not None
            and trff is not None
            and tact > 0
            and trff > 0
        ):
            if not pos.trail_armed and pnl_pct >= tact:
                pos.trail_armed = True
                pos.trail_peak_price = float(current_price)
            if pos.trail_armed and pos.trail_peak_price is not None:
                ep = float(pos.entry_price)
                tp_peak = float(pos.trail_peak_price)
                cr = float(current_price)
                rf = float(trff)
                if pos.side == "long":
                    tp_peak = max(tp_peak, cr)
                    pos.trail_peak_price = tp_peak
                    if tp_peak > ep:
                        give = tp_peak - rf * (tp_peak - ep)
                        if cr <= give and pnl_pct >= float(min_trail_exit_pnl_pct):
                            exit_reason = f"trailing_giveback_{int(rf * 100)}pct_mfe"
                else:
                    tp_peak = min(tp_peak, cr)
                    pos.trail_peak_price = tp_peak
                    if tp_peak < ep:
                        give = tp_peak + rf * (ep - tp_peak)
                        if cr >= give and pnl_pct >= float(min_trail_exit_pnl_pct):
                            exit_reason = f"trailing_giveback_{int(rf * 100)}pct_mfe"

        if exit_reason is None and pnl_pct >= tp:
            exit_reason = f"take_profit_{tp:.4g}pct"

        if exit_reason is None and thesis_ok and regime_would_exit:
            exit_reason = "regime_flip"

        if exit_reason is None and thesis_ok and weak_prob_would_exit:
            exit_reason = "weak_transition_top1_prob"

        if exit_reason is None and thesis_ok and weak_mass_would_exit:
            exit_reason = "weak_transition_label_mass"

        if exit_reason is None and exit_on_plan_absent and plan_id not in active_plan_ids:
            if grace_sec <= 0.0:
                exit_reason = "plan_no_longer_active"
            else:
                if pos.plan_missing_since_utc is None:
                    pos.plan_missing_since_utc = now.isoformat()
                absent_for = (now - _parse_iso_utc(pos.plan_missing_since_utc)).total_seconds()
                if absent_for >= grace_sec:
                    exit_reason = "plan_no_longer_active"

        signal = exit_reason or "hold"
        grace_note = ""
        if (
            exit_reason is None
            and plan_id not in active_plan_ids
            and exit_on_plan_absent
            and grace_sec > 0.0
            and pos.plan_missing_since_utc is not None
        ):
            absent_for = (now - _parse_iso_utc(pos.plan_missing_since_utc)).total_seconds()
            grace_note = f" (plan absent {absent_for:.0f}s / {grace_sec:.0f}s grace)"
        elif exit_reason is None and plan_id not in active_plan_ids and not exit_on_plan_absent:
            grace_note = " (plan off active list — hold; exit on price/thesis only)"
        hold_note = ""
        if exit_reason is None and not thesis_ok and (
            regime_would_exit or weak_prob_would_exit or weak_mass_would_exit
        ):
            hold_note = f" thesis_exit_deferred {age_sec:.0f}s / {mh:.0f}s min_hold"
        trans_note = f" transition_top1_p={p_next:.4f}" if p_next is not None else ""
        rk_note = f" cur_regime={cur_rk[:40]}" if cur_rk else ""
        mass_note = ""
        if aligned_mass is not None:
            mass_note = f" label_aligned_mass({pos.direction})={aligned_mass:.3f}"
        print(
            f"  [equity-monitor] {pos.symbol}: side={pos.side} "
            f"entry={pos.entry_price:.2f} current={current_price:.2f} "
            f"pnl=${pnl:+.2f} ({pnl_pct:+.2f}%) signal={signal}{grace_note}{hold_note}{trans_note}{mass_note}{rk_note}",
            flush=True,
        )

        _append_log(log_path, {
            "timestamp_utc": now.isoformat(),
            "plan_id": plan_id,
            "symbol": pos.symbol,
            "strategy": pos.direction,
            "action": "hold",
            "quantity": pos.quantity,
            "current_mark": current_price,
            "entry_debit": pos.entry_price,
            "order_id": pos.ibkr_order_id or "",
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 4),
            "signal": signal,
            "closed": "0",
            "note": "",
        })

        if exit_reason is None:
            continue

        # Close position
        close_action = "SELL" if pos.side == "long" else "BUY"
        print(f"  [equity] CLOSING {pos.symbol}: {close_action} {pos.quantity} shares — {exit_reason} [dry={dry_run}]", flush=True)

        close_order_id: int | None = None
        if not dry_run and app is not None:
            try:
                close_order_id = app.place_stock_order(pos.symbol, close_action, pos.quantity)
                fill_meta = _coerce_ibkr_fill_meta(app.wait_for_order(close_order_id))
                print(f"    close order_id={close_order_id} fill={fill_meta}", flush=True)
                filled_qty = float(fill_meta.get("filled") or 0.0)
                if filled_qty <= 0.0:
                    raise RuntimeError(
                        f"IBKR close order {close_order_id} returned without fill"
                    )
                avg_px = float(fill_meta.get("avg_fill_price") or 0.0)
                if avg_px > 0.0:
                    current_price = avg_px
                    if pos.side == "long":
                        pnl = (current_price - pos.entry_price) * pos.quantity
                        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100.0
                    else:
                        pnl = (pos.entry_price - current_price) * pos.quantity
                        pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100.0
            except Exception as exc:
                print(f"    [equity] {pos.symbol}: close order error — {exc}", flush=True)
                pos.close_order_id = close_order_id
                _append_log(log_path, {
                    "timestamp_utc": now.isoformat(),
                    "plan_id": plan_id,
                    "symbol": pos.symbol,
                    "strategy": pos.direction,
                    "action": "hold",
                    "quantity": pos.quantity,
                    "current_mark": current_price,
                    "entry_debit": pos.entry_price,
                    "order_id": close_order_id or "",
                    "unrealized_pnl": round(pnl, 2),
                    "unrealized_pnl_pct": round(pnl_pct, 4),
                    "signal": "close_order_error",
                    "closed": "0",
                    "note": f"{exit_reason}: {exc}",
                })
                # Keep local state open so the next tick retries the close.
                continue

        pos.status = "closed"
        pos.exit_price = current_price
        pos.exit_ts = now.isoformat()
        pos.exit_reason = exit_reason
        pos.close_order_id = close_order_id
        pos.plan_missing_since_utc = None
        pos.trail_armed = False
        pos.trail_peak_price = None

        _append_log(log_path, {
            "timestamp_utc": pos.exit_ts,
            "plan_id": plan_id,
            "symbol": pos.symbol,
            "strategy": pos.direction,
            "action": close_action,
            "quantity": pos.quantity,
            "current_mark": current_price,
            "entry_debit": pos.entry_price,
            "order_id": close_order_id or "",
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 4),
            "signal": "closed",
            "closed": "1",
            "note": exit_reason,
        })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR equity paper trade from regime plans")
    parser.add_argument("--plans", default=str(PLANS_PATH), help="Path to universe_trade_plans.json")
    parser.add_argument("--state", default=str(EQUITY_STATE_PATH), help="Path to equity positions state JSON")
    parser.add_argument("--log", default=str(EQUITY_LOG_PATH), help="Path to equity trade log CSV")
    parser.add_argument("--position-usd", type=float, default=10_000.0,
                        help="Target notional USD per position (default: $10,000; ignored if --risk-usd is set)")
    parser.add_argument("--risk-usd", type=float, default=None,
                        help="Dollar amount to risk per trade (e.g. 500). If set, overrides --position-usd.")
    parser.add_argument("--use-account-scale", action="store_true",
                        help="Scale position size based on account balance (NLV).")
    parser.add_argument("--max-account-pct", type=float, default=10.0,
                        help="Max percentage of account balance per position (default: 10.0)")
    parser.add_argument("--stop-pct", type=float, default=5.0,
                        help="Hard stop loss %% below entry (default: 5)")
    parser.add_argument("--target-pct", type=float, default=10.0,
                        help="Take-profit %% above entry (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Paper-mode without IBKR orders (log only)")
    parser.add_argument("--monitor-only", action="store_true",
                        help="Skip opening new positions; only evaluate existing ones")
    parser.add_argument(
        "--plan-missing-grace-sec",
        type=float,
        default=None,
        help=(
            "Seconds a plan may be absent from the active universe before equity auto-close "
            "(default: env RLM_EQUITY_PLAN_MISSING_GRACE_SEC or 900). Use 0 for immediate close."
        ),
    )
    parser.add_argument(
        "--min-most-likely-next-prob",
        type=float,
        default=None,
        help=(
            "Exit stock leg if calibrated/top-1 next-step regime transition probability falls "
            "below this threshold (unset = disabled; overrides env "
            "RLM_EQUITY_MIN_MOST_LIKELY_NEXT_PROB)."
        ),
    )
    parser.add_argument(
        "--min-next-label-aligned-mass",
        type=float,
        default=None,
        help=(
            "Exit when Σ P(next state)×label-weight for the traded direction falls below this "
            "(0–1; unset = disabled; overrides RLM_EQUITY_MIN_NEXT_LABEL_ALIGNED_MASS). "
            "Uses full transition row + HMM/Markov state labels from the universe pipeline."
        ),
    )
    parser.add_argument(
        "--exit-on-plan-absent",
        action="store_true",
        help=(
            "Close stock when plan_id is not in the active universe (after grace). "
            "Default on (disable with --no-exit-on-plan-absent or RLM_EQUITY_EXIT_ON_PLAN_ABSENT=0)."
        ),
    )
    parser.add_argument(
        "--no-exit-on-plan-absent",
        action="store_true",
        help="Keep equities when universe JSON drops the plan (legacy hold behavior).",
    )
    parser.add_argument(
        "--min-hold-thesis-sec",
        type=float,
        default=None,
        help=(
            "Minimum seconds in trade before regime/transition-model exits may fire "
            "(default: env RLM_EQUITY_MIN_HOLD_THESIS_SEC or 1800). Hard stop / trail / TP always apply."
        ),
    )
    parser.add_argument(
        "--trail-activate-pct",
        type=float,
        default=None,
        help=(
            "Arm MFE trailing give-back once unrealized PnL%% reaches this "
            "(default: env RLM_EQUITY_TRAIL_ACTIVATE_PCT or 8; 0 or negative disables)."
        ),
    )
    parser.add_argument(
        "--trail-retrace-frac",
        type=float,
        default=None,
        help=(
            "Fraction of peak favorable excursion from entry to give back before trail exit "
            "(default: env RLM_EQUITY_TRAIL_RETRACE_FRAC or 0.25)."
        ),
    )
    parser.add_argument(
        "--min-trail-exit-pct",
        type=float,
        default=None,
        help=(
            "Trail exit only if unrealized PnL%% is still at/above this (default 0 = breakeven)."
        ),
    )
    dhg = parser.add_mutually_exclusive_group()
    dhg.add_argument(
        "--dynamic-horizon",
        action="store_true",
        help=(
            "Blend stop/target/min-hold/trail from each plan's context (bar_size, TF confirmation, "
            "transition stability, size_fraction). Overrides env to ON for this run."
        ),
    )
    dhg.add_argument(
        "--no-dynamic-horizon",
        action="store_true",
        help="Use one global policy for all opens (overrides env).",
    )
    args = parser.parse_args()
    if args.dynamic_horizon:
        dh_flag: bool | None = True
    elif args.no_dynamic_horizon:
        dh_flag = False
    else:
        dh_flag = None
    dynamic_h = _dynamic_horizon_enabled(dh_flag)

    def _resolve_data_path(raw: str | Path) -> Path:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (DATA_ROOT / p)

    plans_path = _resolve_data_path(args.plans)
    state_path = _resolve_data_path(args.state)
    log_path = _resolve_data_path(args.log)
    _ensure_equity_log_file(log_path)

    print(f"\n{'='*60}", flush=True)
    print(f"  IBKR Equity Paper Trade  |  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", flush=True)
    print(f"  plans       : {plans_path}", flush=True)
    print(f"  position_usd: ${args.position_usd:,.0f}{' (ignored)' if args.risk_usd or args.use_account_scale else ''}", flush=True)
    if args.risk_usd:
        print(f"  risk_usd    : ${args.risk_usd:,.0f}", flush=True)
    if args.use_account_scale:
        print(f"  account_scale: max {args.max_account_pct}% of NLV scaled by confidence", flush=True)
    grace_sec = _plan_missing_grace_sec(args.plan_missing_grace_sec)
    min_np = _min_most_likely_next_prob(args.min_most_likely_next_prob)
    min_lm = _min_next_label_aligned_mass(args.min_next_label_aligned_mass)
    exit_on_plan_absent = _exit_on_plan_absent(
        False if args.no_exit_on_plan_absent else (True if args.exit_on_plan_absent else None)
    )
    min_hold_main = _min_hold_thesis_sec(args.min_hold_thesis_sec)
    trail_act = _trail_activate_pct(args.trail_activate_pct)
    trail_rf = _trail_retrace_frac(args.trail_retrace_frac)
    min_trail_exit = _min_trail_exit_pnl_pct(args.min_trail_exit_pct)
    print(f"  stop / target : -{args.stop_pct}% / +{args.target_pct}% (baseline when dynamic off)", flush=True)
    print(f"  dynamic horizon: {dynamic_h}", flush=True)
    print(f"  exit on plan absent: {exit_on_plan_absent}", flush=True)
    if exit_on_plan_absent:
        print(f"  plan missing grace (universe): {grace_sec:.0f}s", flush=True)
    print(f"  min hold (thesis exits): {min_hold_main:.0f}s", flush=True)
    if trail_act is not None and trail_rf is not None:
        print(f"  trailing give-back: arm >={trail_act:.2f}% MFE, retrace {trail_rf:.0%}", flush=True)
    else:
        print("  trailing give-back: disabled", flush=True)
    print(
        f"  min transition top-1 p: {'%.4f (active)' % min_np if min_np else 'disabled'}",
        flush=True,
    )
    print(
        f"  min label-aligned mass: {'%.4f (active)' % min_lm if min_lm else 'disabled'}",
        flush=True,
    )
    print(f"  dry_run     : {args.dry_run}", flush=True)
    print(f"{'='*60}\n", flush=True)

    plans = _load_plans(plans_path)
    positions = _load_state(state_path)
    active_plan_ids = {p["plan_id"] for p in plans if p.get("plan_id")}
    plan_by_id = _plans_by_plan_id(plans)
    _merge_universe_rows_for_open_positions(plan_by_id, plans_path, positions)

    account_nlv: float | None = None
    if args.use_account_scale:
        try:
            print("[equity] Fetching account balance for scaling …", flush=True)
            snap = fetch_ibkr_account_snapshot(timeout_sec=30.0)
            for row in snap.account_summary:
                if row.tag == "NetLiquidation":
                    try:
                        account_nlv = float(row.value)
                        print(f"  [equity] Account NLV: ${account_nlv:,.2f}", flush=True)
                        break
                    except (ValueError, TypeError):
                        pass
            if account_nlv is None:
                print("  [equity] [warn] Could not find NetLiquidation in account summary. Falling back to default sizing.", flush=True)
        except Exception as e:
            print(f"  [equity] [error] Could not fetch account snapshot: {e}. Falling back to default sizing.", flush=True)

    print(f"[equity] Loaded {len(plans)} active plans, {len(positions)} tracked positions", flush=True)

    from rlm.utils.market_hours import entry_window_open, is_rth_now, session_label

    equity_rth_only = (os.environ.get("RLM_EQUITY_RTH_ONLY") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if equity_rth_only and not is_rth_now():
        print(
            f"[equity] outside NYSE RTH ({session_label()}) — no entries or exits this tick",
            flush=True,
        )
        _save_state(positions, state_path)
        return
    if equity_rth_only and not args.monitor_only and not entry_window_open():
        print(
            f"[equity] outside entry window ({session_label()}) — monitor/exits only, no new stock opens",
            flush=True,
        )
        args.monitor_only = True

    if args.dry_run:
        # No IBKR connection needed — work in dry-run mode
        if not args.monitor_only:
            open_equity_positions(
                plans=plans, positions=positions, position_usd=args.position_usd,
                risk_usd=args.risk_usd, stop_pct=args.stop_pct,
                target_pct=args.target_pct,
                min_hold_sec=min_hold_main,
                trail_activate_pct=trail_act,
                trail_retrace_frac=trail_rf,
                dynamic_horizon=dynamic_h,
                account_nlv=account_nlv, max_account_pct=args.max_account_pct,
                dry_run=True, app=None, plans_path=plans_path, log_path=log_path,
            )
        evaluate_equity_positions(
            positions=positions,
            active_plan_ids=active_plan_ids,
            plan_by_id=plan_by_id,
            stop_pct=args.stop_pct,
            target_pct=args.target_pct,
            grace_sec=grace_sec,
            min_most_likely_next_prob=min_np,
            min_next_label_aligned_mass=min_lm,
            dry_run=True,
            app=None,
            log_path=log_path,
            exit_on_plan_absent=exit_on_plan_absent,
            min_hold_sec=min_hold_main,
            trail_activate_pct=trail_act,
            trail_retrace_frac=trail_rf,
            min_trail_exit_pnl_pct=min_trail_exit,
        )
        _save_state(positions, state_path)
        print("\n[equity] dry-run complete.", flush=True)
        return

    # Live IBKR connection
    with ibkr_equity_connection() as app:
        if not args.monitor_only:
            open_equity_positions(
                plans=plans, positions=positions, position_usd=args.position_usd,
                risk_usd=args.risk_usd, stop_pct=args.stop_pct,
                target_pct=args.target_pct,
                min_hold_sec=min_hold_main,
                trail_activate_pct=trail_act,
                trail_retrace_frac=trail_rf,
                dynamic_horizon=dynamic_h,
                account_nlv=account_nlv, max_account_pct=args.max_account_pct,
                dry_run=False, app=app, plans_path=plans_path, log_path=log_path,
            )
        evaluate_equity_positions(
            positions=positions,
            active_plan_ids=active_plan_ids,
            plan_by_id=plan_by_id,
            stop_pct=args.stop_pct,
            target_pct=args.target_pct,
            grace_sec=grace_sec,
            min_most_likely_next_prob=min_np,
            min_next_label_aligned_mass=min_lm,
            dry_run=False,
            app=app,
            log_path=log_path,
            exit_on_plan_absent=exit_on_plan_absent,
            min_hold_sec=min_hold_main,
            trail_activate_pct=trail_act,
            trail_retrace_frac=trail_rf,
            min_trail_exit_pnl_pct=min_trail_exit,
        )

    _save_state(positions, state_path)
    print("\n[equity] done.", flush=True)


if __name__ == "__main__":
    main()
