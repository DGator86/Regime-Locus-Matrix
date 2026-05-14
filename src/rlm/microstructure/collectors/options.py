"""Shim: ``python -m rlm.microstructure.collectors.options``."""

from rlm.data.microstructure.collectors.options import OptionsCollector, _main

__all__ = ["OptionsCollector"]


if __name__ == "__main__":  # pragma: no cover - exercised via CLI smoke test
    _main()
