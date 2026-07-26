"""Regression: concurrent challenge sessions must not lose accepted opens."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.engine import ChallengeEngine
from rlm.challenge.tracker import ChallengeTracker


def _mp_run_session(data_root: str, tag: str, queue: mp.Queue, *, suppress_exits: bool) -> None:
    cfg = replace(
        ChallengeConfig(seed_capital=1_000.0, target_capital=25_000.0),
        max_concurrent_positions=1,
        stage1_size_frac=0.85,
    )
    tracker = ChallengeTracker(data_root=data_root)
    eng = ChallengeEngine(cfg, tracker)
    if suppress_exits:
        with patch.object(ChallengeEngine, "_evaluate_position", return_value=None):
            summary = eng.run_session(
                "long",
                underlying_price=500.0,
                signal_alignment=0.95,
                confidence=0.95,
                iv=0.25,
                session_date="2026-07-26",
                regime_key="trend",
            )
    else:
        summary = eng.run_session(
            "long",
            underlying_price=500.0,
            signal_alignment=0.95,
            confidence=0.95,
            iv=0.25,
            session_date="2026-07-26",
            regime_key="trend",
        )
    queue.put((tag, summary.new_position is not None, summary.balance_after))


def test_cross_process_run_session_preserves_session_count(tmp_path: Path) -> None:
    """Overlapping challenge --run processes must not last-writer-wins drop a session.

    Concrete trigger: ``run_everything`` leaves a challenge subprocess after
    ``RLM_CHALLENGE_JOIN_SEC`` while a periodic tick (or ``rlm-challenge-loop``)
    starts another ``rlm challenge --run`` against the same state.json.
    """
    cfg = replace(
        ChallengeConfig(seed_capital=1_000.0, target_capital=25_000.0),
        max_concurrent_positions=1,
        stage1_size_frac=0.85,
    )
    tracker = ChallengeTracker(data_root=str(tmp_path))
    tracker.reset(cfg)

    queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(
            target=_mp_run_session,
            args=(str(tmp_path), "p0", queue),
            kwargs={"suppress_exits": False},
        ),
        mp.Process(
            target=_mp_run_session,
            args=(str(tmp_path), "p1", queue),
            kwargs={"suppress_exits": False},
        ),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    results = [queue.get(timeout=5) for _ in range(2)]
    assert len(results) == 2
    final = tracker.load()
    # Without the exclusive lock, both processes loaded the empty book and the
    # slower save wiped the faster session (session_count stayed 1).
    assert final.session_count == 2
    assert len(final.open_positions) <= cfg.max_concurrent_positions


def test_cross_process_empty_book_entry_race(tmp_path: Path) -> None:
    """With exits suppressed, only one overlapping process may open from empty."""
    cfg = replace(
        ChallengeConfig(seed_capital=1_000.0, target_capital=25_000.0),
        max_concurrent_positions=1,
        stage1_size_frac=0.85,
    )
    tracker = ChallengeTracker(data_root=str(tmp_path))
    tracker.reset(cfg)

    queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(
            target=_mp_run_session,
            args=(str(tmp_path), "p0", queue),
            kwargs={"suppress_exits": True},
        ),
        mp.Process(
            target=_mp_run_session,
            args=(str(tmp_path), "p1", queue),
            kwargs={"suppress_exits": True},
        ),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    results = [queue.get(timeout=5) for _ in range(2)]
    opened_claims = sum(1 for _, opened, _ in results if opened)
    final = tracker.load()
    assert opened_claims == 1
    assert len(final.open_positions) == 1
    assert final.session_count == 2
    assert final.balance < cfg.seed_capital


def test_save_is_atomic_replace(tmp_path: Path) -> None:
    cfg = ChallengeConfig(seed_capital=1_000.0, target_capital=25_000.0)
    tracker = ChallengeTracker(data_root=str(tmp_path))
    state = tracker.reset(cfg)
    state.balance = 1_234.0
    tracker.save(state)
    assert tracker.state_path().is_file()
    assert not list(tracker.state_path().parent.glob(".state.json.*.tmp"))
    assert tracker.load().balance == 1_234.0
