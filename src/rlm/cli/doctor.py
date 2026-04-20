"""``rlm doctor`` — preflight environment, provider, and data lake diagnostics."""

from __future__ import annotations

import argparse

from rlm.cli.common import add_data_root_arg
from rlm.core.services.diagnostics_service import DiagnosticsService
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm doctor",
        description=(
            "Diagnose environment, dependencies, providers, and data lake. "
            "Exits with code 1 if any required check fails."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Show detail for every check")
    add_data_root_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log.info("doctor start  data_root=%s", args.data_root or "(default)")

    report = DiagnosticsService().run(verbose=args.verbose, data_root=args.data_root)

    status_icon = {True: "[OK]  ", False: "[FAIL]", None: "[SKIP]"}

    print("\nRLM Doctor Report")
    print("=" * 60)
    for check in report.checks:
        icon = status_icon.get(check.passed, "[?]   ")
        print(f"  {icon}  {check.name}")
        if check.detail and (args.verbose or not check.passed):
            print(f"         {check.detail}")

    passed = sum(1 for c in report.checks if c.passed is True)
    skipped = sum(1 for c in report.checks if c.passed is None)
    failed = sum(1 for c in report.checks if c.passed is False)
    total = len(report.checks)

    print(f"\n  {passed} passed  •  {skipped} skipped  •  {failed} failed  (of {total})")

    if not report.all_passed:
        print("\n  Run with -v for details on each failed check.")
        raise SystemExit(1)

    print("\n  All required checks passed.")
