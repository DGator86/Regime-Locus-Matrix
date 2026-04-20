"""``rlm doctor`` — diagnose the environment, providers, and data lake."""

from __future__ import annotations

import argparse

from rlm.core.services.diagnostics_service import DiagnosticsService
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm doctor",
        description="Diagnose environment, dependencies, providers, and data lake.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log.info("Running RLM diagnostics...")

    report = DiagnosticsService().run(verbose=args.verbose)

    status_icon = {True: "[OK]", False: "[FAIL]", None: "[SKIP]"}

    print("\nRLM Doctor Report")
    print("=" * 50)
    for check in report.checks:
        icon = status_icon.get(check.passed, "[?]")
        print(f"  {icon}  {check.name}")
        if check.detail and (args.verbose or not check.passed):
            print(f"       {check.detail}")

    passed = sum(1 for c in report.checks if c.passed)
    total = len(report.checks)
    print(f"\n{passed}/{total} checks passed")

    if not report.all_passed:
        raise SystemExit(1)
