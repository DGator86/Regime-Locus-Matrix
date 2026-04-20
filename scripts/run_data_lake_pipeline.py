"""Legacy wrapper — backwards compatibility shim for ``rlm ingest``.

Requires the package to be installed:

    pip install -e .

Preferred usage (works from any directory after install):

    rlm ingest --symbol SPY [options]

This wrapper will be removed in a future release.
"""

from rlm.cli.ingest import main

if __name__ == "__main__":
    main()
