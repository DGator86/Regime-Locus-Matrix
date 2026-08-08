"""Refresh open challenge positions with live underlying and option mids."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from rlm.challenge.config import ChallengeConfig, apply_challenge_profile_env
from rlm.challenge.pricing import updated_premium
from rlm.challenge.state import ChallengePosition, ChallengeState
from rlm.challenge.tracker import ChallengeTracker
from rlm.market.live_quotes import fetch_equity_quote, fetch_option_mid_per_share, live_marks_enabled


def _days_between(start: str, end: str) -> int:
    try:
        d1 = date.fromisoformat(start)
        d2 = date.fromisoformat(end)
        return max(0, (d2 - d1).days)
    except ValueError:
        return 1


def option_expiry_date(entry_date: str, dte_at_entry: int) -> date:
    start = date.fromisoformat(str(entry_date)[:10])
    return start + timedelta(days=int(dte_at_entry))


def mark_open_position_premium(
    pos: ChallengePosition,
    cfg: ChallengeConfig,
    *,
    session_date: str,
    pipeline_underlying: float,
    iv: float | None = None,
) -> tuple[float, float, int]:
    """Mark one leg for exit checks. Returns (premium/sh, underlying used, dte remaining)."""
    days_elapsed = _days_between(pos.entry_date, session_date)
    new_dte = max(0, pos.dte_at_entry - days_elapsed)
    iv_use = iv if iv is not None else pos.iv_at_entry
    underlying = float(pipeline_underlying) if pipeline_underlying and pipeline_underlying > 0 else 0.0
    new_premium: float

    quote = fetch_equity_quote(pos.symbol) if live_marks_enabled() else None
    if quote is not None and quote.price > 0:
        underlying = quote.price
        exp = option_expiry_date(pos.entry_date, pos.dte_at_entry)
        live_mid = fetch_option_mid_per_share(
            pos.symbol,
            strike=float(pos.strike),
            expiry=exp,
            option_type=pos.option_type,
        )
        if live_mid is not None and live_mid > 0:
            return live_mid, underlying, new_dte

    # Synthetic mark needs a positive underlying; fall back to entry spot when
    # live/pipeline quotes are missing so Telegram /positions and --status do
    # not ValueError inside updated_premium (log(0) / divide-by-zero).
    if underlying <= 0:
        underlying = float(pos.underlying_entry) if pos.underlying_entry > 0 else 0.0
    if underlying <= 0:
        return max(float(pos.current_premium or pos.premium_per_share), 0.01), underlying, new_dte

    new_premium = updated_premium(
        entry_premium=pos.premium_per_share,
        delta=pos.delta_at_entry,
        underlying_entry=pos.underlying_entry if pos.underlying_entry > 0 else underlying,
        underlying_now=underlying,
        days_elapsed=days_elapsed,
        dte_remaining=new_dte,
        iv=iv_use,
    )
    return new_premium, underlying, new_dte


def refresh_challenge_state(
    state: ChallengeState,
    cfg: ChallengeConfig,
    *,
    session_date: str | None = None,
) -> str | None:
    """Update ``current_premium`` / PnL on open legs. Returns quote as-of UTC or None."""
    if not state.open_positions:
        return None

    session_date = session_date or date.today().isoformat()
    quote = fetch_equity_quote(cfg.symbol)
    asof = quote.asof_utc if quote is not None else None
    pipe_px = quote.price if quote is not None and quote.price > 0 else 0.0

    for pos in state.open_positions:
        new_premium, _, new_dte = mark_open_position_premium(
            pos,
            cfg,
            session_date=session_date,
            pipeline_underlying=pipe_px,
        )
        pos.dte_remaining = new_dte
        pos.current_premium = new_premium
        pos.current_value = new_premium * pos.qty * 100
        pos.unrealised_pnl = pos.current_value - pos.total_cost
        mult = new_premium / pos.premium_per_share if pos.premium_per_share > 0 else 1.0
        if mult > pos.peak_premium_mult:
            pos.peak_premium_mult = mult

    state.last_updated = datetime.now(tz=timezone.utc).isoformat()
    return asof


def refresh_challenge_at_root(root: Path) -> str | None:
    """Load state, refresh marks, persist. No-op if no state file."""
    ch_path = root / "data" / "challenge" / "state.json"
    if not ch_path.is_file():
        return None
    tracker = ChallengeTracker(data_root=str(root / "data"))
    try:
        state = tracker.load()
    except FileNotFoundError:
        return None
    cfg = apply_challenge_profile_env(ChallengeConfig())
    asof = refresh_challenge_state(state, cfg)
    if asof is not None:
        tracker.save(state)
    return asof
