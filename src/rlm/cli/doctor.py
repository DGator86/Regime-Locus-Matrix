"""``rlm doctor`` — preflight environment, provider, and data lake diagnostics.

Suitable for use in CI:

    rlm doctor --json        # machine-readable JSON output
    rlm doctor --strict      # exit 1 only on hard failures (not optional checks)
    rlm doctor --symbol SPY  # also validate that data exists for a symbol
"""

from __future__ import annotations

import argparse
import json
import sys

from rlm.cli.common import add_backend_arg, add_data_root_arg, add_profile_args
from rlm.core.services.diagnostics_service import DiagnosticsService
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm doctor",
        description=(
            "Diagnose environment, dependencies, providers, and data lake. "
            "Exits with code 1 if required checks fail."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Show detail for every check")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output results as JSON")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 only on hard failures (FAIL); ignore SKIP/optional",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="Also check that expected data exists for this symbol (e.g. SPY)",
    )
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log.info("doctor start  data_root=%s backend=%s", args.data_root or "(default)", args.backend)

    report = DiagnosticsService().run(
        verbose=args.verbose,
        data_root=args.data_root,
        backend=args.backend,
        profile=getattr(args, "profile", None),
        config_path=getattr(args, "config", None),
        symbol=args.symbol,
    )

    if args.json_output:
        _print_json(report)
    else:
        _print_human(report, args.verbose)

    passed = sum(1 for c in report.checks if c.passed is True)
    skipped = sum(1 for c in report.checks if c.passed is None)
    failed = sum(1 for c in report.checks if c.passed is False)

    if args.strict:
        if failed > 0:
            raise SystemExit(1)
    else:
        if not report.all_passed:
            raise SystemExit(1)


def _print_human(report: "DiagnosticsReport", verbose: bool) -> None:  # type: ignore[name-defined]
    status_icon = {True: "[OK]  ", False: "[FAIL]", None: "[SKIP]"}
    print("\nRLM Doctor Report")
    print("=" * 60)
    for check in report.checks:
        icon = status_icon.get(check.passed, "[?]   ")
        print(f"  {icon}  {check.name}")
        if check.detail and (verbose or not check.passed):
            print(f"         {check.detail}")

    passed = sum(1 for c in report.checks if c.passed is True)
    skipped = sum(1 for c in report.checks if c.passed is None)
    failed = sum(1 for c in report.checks if c.passed is False)
    total = len(report.checks)
    print(f"\n  {passed} passed  •  {skipped} skipped  •  {failed} failed  (of {total})")

    if not report.all_passed:
        print("\n  Run with -v for details on failed checks.")
    else:
        print("\n  All required checks passed.")


def _print_json(report: "DiagnosticsReport") -> None:  # type: ignore[name-defined]
    data = {
        "all_passed": report.all_passed,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "detail": c.detail,
            }
            for c in report.checks
        ],
        "summary": {
            "passed": sum(1 for c in report.checks if c.passed is True),
            "skipped": sum(1 for c in report.checks if c.passed is None),
            "failed": sum(1 for c in report.checks if c.passed is False),
            "total": len(report.checks),
        },
    }
    print(json.dumps(data, indent=2))
