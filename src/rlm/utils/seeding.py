"""
Deterministic pipeline seeding.

Call seed_everything(42) before any backtest, walk-forward run, or Monte Carlo
simulation to guarantee reproducible results across runs. Without this, small
differences in HMM sampling, Kronos temperature sampling, or numpy RNG state
make it impossible to diff two runs or trust that an improvement is real.

Usage:
    from rlm.utils.seeding import seed_everything
    seed_everything(42)
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """
    Seed all RNG sources used by the pipeline.

    Covers: Python random, NumPy. PyTorch and CUDA are seeded when available
    (Kronos foundation model). Statsmodels uses NumPy's global RNG so is
    covered by the numpy seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
