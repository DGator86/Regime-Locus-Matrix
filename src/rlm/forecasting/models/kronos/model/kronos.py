"""Torch Kronos model entrypoints.

Default behavior is a lightweight stub for tests. To enable full Kronos runtime
without modifying this repository, set ``RLM_KRONOS_VENDOR_PATH`` to a checkout
of https://github.com/DGator86/Kronos (the directory that contains ``model/``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_vendor_module() -> ModuleType | None:
    root = (os.environ.get("RLM_KRONOS_VENDOR_PATH") or "").strip()
    if not root:
        return None
    vendor_root = Path(root).expanduser().resolve()
    kronos_path = vendor_root / "model" / "kronos.py"
    if not kronos_path.is_file():
        return None
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    spec = importlib.util.spec_from_file_location("rlm_kronos_vendor_runtime", str(kronos_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VENDOR = _load_vendor_module()
if _VENDOR is not None and all(hasattr(_VENDOR, name) for name in ("KronosTokenizer", "Kronos", "KronosPredictor")):
    KronosTokenizer = _VENDOR.KronosTokenizer  # type: ignore[misc,assignment]
    Kronos = _VENDOR.Kronos  # type: ignore[misc,assignment]
    KronosPredictor = _VENDOR.KronosPredictor  # type: ignore[misc,assignment]
else:

    class KronosTokenizer:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "KronosTokenizer":
            raise ImportError(
                "KronosTokenizer.from_pretrained is not available in the stub module. "
                "Set RLM_KRONOS_VENDOR_PATH to a Kronos checkout, or use "
                "KronosForecastPipeline(predictor=mock) in tests."
            )


    class Kronos:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "Kronos":
            raise ImportError(
                "Kronos.from_pretrained is not available in the stub module. "
                "Set RLM_KRONOS_VENDOR_PATH to a Kronos checkout, or use "
                "KronosForecastPipeline(predictor=mock) in tests."
            )


    class KronosPredictor:
        """Torch predictor with ``predict(df, x_timestamp, y_timestamp, ...) -> DataFrame``."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError(
                "KronosPredictor is a stub unless a Kronos runtime is available via "
                "RLM_KRONOS_VENDOR_PATH."
            )

        def predict(self, **_kwargs: Any):  # pragma: no cover - stub
            raise ImportError("KronosPredictor.predict requires the vendored Kronos runtime.")
