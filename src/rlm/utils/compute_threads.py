"""Limit BLAS/OpenMP and PyTorch threads for shared VPS / multi-tenant hosts.

NumPy/SciPy/Pandas often link to OpenBLAS or MKL, which default to using every
logical CPU.  A single ``run_universe_options_pipeline`` or ``rlm forecast``
process can then pin the host at ~100% × core count.  This module sets
conservative caps **before** importing NumPy when callers invoke it early
enough.

Environment
-----------
``RLM_MAX_CPU_THREADS``:
    Positive integer caps BLAS env vars and ``torch.set_num_threads``.  If
    unset, a default of ``max(1, min(4, (cpu_count or 2) // 2))`` is used when
    no BLAS-related env vars are already set.

If any of ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
``NUMEXPR_NUM_THREADS``, or ``VECLIB_MAXIMUM_THREADS`` is already set, BLAS
variables are not overwritten (operators can enforce site-wide policy).  PyTorch
intra-op threads are still capped to the resolved value when Torch is
available.
"""

from __future__ import annotations

import os
from typing import Any

_BLAS_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def default_thread_cap() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(4, max(1, cpu // 2)))


def resolve_thread_cap(explicit: int | None = None) -> int:
    raw = os.environ.get("RLM_MAX_CPU_THREADS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    if explicit is not None:
        return max(1, int(explicit))
    return default_thread_cap()


def apply_compute_thread_env(
    max_threads: int | None = None,
    *,
    force_blas_env: bool = False,
) -> dict[str, Any]:
    """Apply thread caps; call before ``import numpy`` / ``import torch`` when possible.

    Parameters
    ----------
    max_threads
        Override cap (minimum 1).  Ignored if ``RLM_MAX_CPU_THREADS`` is set.
    force_blas_env
        If True, set BLAS-related env vars even when the user already set one
        (use only for tests or controlled subprocesses).
    """
    n = resolve_thread_cap(max_threads)
    report: dict[str, Any] = {"effective_cap": n}
    user_set_any = any(os.environ.get(k) for k in _BLAS_ENV_NAMES)
    if force_blas_env or not user_set_any:
        for k in _BLAS_ENV_NAMES:
            if force_blas_env or not os.environ.get(k):
                os.environ[k] = str(n)
        report["blas_env"] = "set"
    else:
        report["blas_env"] = "skipped_existing"

    try:
        import torch

        torch.set_num_threads(n)
        inter = max(1, min(2, n))
        try:
            torch.set_num_interop_threads(inter)
        except RuntimeError:
            report["torch_interop"] = "unchanged_after_init"
        else:
            report["torch_interop"] = inter
        report["torch_intra"] = n
    except ImportError:
        report["torch"] = "not_installed"

    return report
