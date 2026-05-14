"""Shim: ``python -m rlm.microstructure.collectors.underlying``."""

from rlm.data.microstructure.collectors.underlying import UnderlyingCollector, _main

__all__ = ["UnderlyingCollector"]


if __name__ == "__main__":  # pragma: no cover - exercised via CLI smoke test
    _main()
