#!/usr/bin/env python3
"""
Launch the RLM Control Center (Streamlit).

From repo root::

    python scripts/run_control_center.py
    python scripts/run_control_center.py --public --port 8501   # VPS / LAN

``--public`` binds ``0.0.0.0`` so you can open ``http://<host>:<port>/`` from another
machine. Default uses Streamlit config (typically ``127.0.0.1`` only).

Requires: ``pip install -e ".[ui]"``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "scripts" / "rlm_control_center" / "app.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--public",
        action="store_true",
        help="Listen on 0.0.0.0 (reachable from LAN / VPS public IP). Default: use .streamlit/config.toml address.",
    )
    ap.add_argument("--port", type=int, default=8501, help="TCP port (default: 8501)")
    args = ap.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.port",
        str(args.port),
    ]
    if args.public:
        cmd.extend(["--server.address", "0.0.0.0"])

    return int(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
