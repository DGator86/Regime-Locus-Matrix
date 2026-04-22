"""Legacy wrapper — backwards compatibility shim for ``rlm forecast``.

Requires the package to be installed:

    pip install -e .

Preferred usage (works from any directory after install):

    rlm forecast --symbol SPY [options]

This wrapper will be removed in a future release.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from rlm.utils.compute_threads import apply_compute_thread_env

apply_compute_thread_env()

from rlm.cli.forecast import main

if __name__ == "__main__":
    main()
