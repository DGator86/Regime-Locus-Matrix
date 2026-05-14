#!/usr/bin/env python3
"""Exit 0 if Hermes + RLM tool registration imports succeed (same path order as run_crew)."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT))
os.environ.setdefault("RLM_ROOT", str(ROOT))

try:
    import run_agent  # noqa: F401
    import rlm_hermes_tools.register_rlm_tools  # noqa: F401
except Exception:
    traceback.print_exc()
    sys.exit(1)
print("smoke_hermes_imports: OK")
