"""Thin wrapper — delegates to ``rlm ingest``.

Kept for backwards compatibility. New usage: ``rlm ingest [options]``
"""

from rlm.cli.ingest import main

if __name__ == "__main__":
    main()
