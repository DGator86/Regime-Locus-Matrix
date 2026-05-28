"""Refresh open challenge positions with live underlying and option mids."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from rlm.challenge.config import ChallengeConfig, apply_challenge_profile_env
from rlm.challenge.engine import _days_between
from rlm.challenge.pricing import updated_premium
from rlm.challenge.state import ChallengeState
from rlm.challenge.tracker import ChallengeTracker
from rlm.market.live_quotes import fetch_equity_quote, fetch_option_mid_per_share


def option_expiry_date(entry_date: str, dte_at_entry: int) -> date:
    start = date.fromisoformat(str(entry_date)[:10])
    return start + timedelta(days=int(dte_at_entry))


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
    if quote is None:
        return None

    asof = quote.asof_utc
    for pos in state.open_positions:
        days_elapsed = _days_between(pos.entry_date, session_date)
        new_dte = max(0, pos.dte_at_entry - days_elapsed)
        exp = option_expiry_date(pos.entry_date, pos.dte_at_entry)

        live_mid = fetch_option_mid_per_share(
            pos.symbol,
            strike=float(pos.strike),
            expiry=exp,
            option_type=pos.option_type,
        )
        if live_mid is not None and live_mid > 0:
            new_premium = live_mid
        else:
            new_premium = updated_premium(
                entry_premium=pos.premium_per_share,
                delta=pos.delta_at_entry,
                underlying_entry=pos.underlying_entry,
                underlying_now=quote.price,
                days_elapsed=days_elapsed,
                dte_remaining=new_dte,
                iv=pos.iv_at_entry,
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
