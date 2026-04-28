"""Thread env helper for BLAS / torch caps."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def clean_blas_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RLM_MAX_CPU_THREADS",
    ):
        monkeypatch.delenv(k, raising=False)


def test_apply_sets_blas_when_unset(clean_blas_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    import importlib

    ct = importlib.import_module("rlm.utils.compute_threads")
    importlib.reload(ct)
    report = ct.apply_compute_thread_env(force_blas_env=True)
    assert report["blas_env"] == "set"
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert report["effective_cap"] == 2


def test_rlm_max_cpu_threads_override(
    clean_blas_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLM_MAX_CPU_THREADS", "3")
    import importlib

    ct = importlib.reload(importlib.import_module("rlm.utils.compute_threads"))
    report = ct.apply_compute_thread_env(force_blas_env=True)
    assert report["effective_cap"] == 3
    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_skips_blas_when_user_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    import importlib

    ct = importlib.reload(importlib.import_module("rlm.utils.compute_threads"))
    report = ct.apply_compute_thread_env()
    assert report["blas_env"] == "skipped_existing"
    assert os.environ["OMP_NUM_THREADS"] == "16"
