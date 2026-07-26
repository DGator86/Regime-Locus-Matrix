"""Regression: concurrent challenge sessions must not lose accepted opens."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.engine import ChallengeEngine
from rlm.challenge.tracker import ChallengeTracker


def test_concurrent_run_session_serializes_opens(tmp_path: Path) -> None:
    """Overlapping run_session calls must not last-writer-wins drop an open.

    Concrete trigger: ``run_everything`` leaves a challenge thread after
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

    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []

    def run_one(tag: str) -> None:
        try:
            # Stagger slightly so both contend for the exclusive lock while
            # still exercising overlapping critical sections.
            if tag == "t1":
                time.sleep(0.02)
            eng = ChallengeEngine(cfg, tracker)
            summary = eng.run_session(
                "long",
                underlying_price=500.0,
                signal_alignment=0.95,
                confidence=0.95,
                iv=0.25,
                session_date="2026-07-26",
                regime_key="trend",
            )
            results.append((tag, summary.new_position is not None))
        except BaseException as exc:  # pragma: no cover - surfacing race failures
            errors.append(exc)

    t0 = threading.Thread(target=run_one, args=("t0",))
    t1 = threading.Thread(target=run_one, args=("t1",))
    t0.start()
    t1.start()
    t0.join(timeout=30)
    t1.join(timeout=30)

    assert not errors, f"concurrent sessions raised: {errors!r}"
    assert len(results) == 2

    final = tracker.load()
    opened_claims = sum(1 for _, opened in results if opened)
    # With max_concurrent=1, at most one session may accept a new open; the
    # other must observe the filled slot after the lock serializes.
    assert opened_claims == 1
    assert len(final.open_positions) == 1
    assert final.session_count == 2
    # Balance reflects exactly one entry debit from seed.
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
