"""``rlm doctor`` — preflight diagnostics."""

from __future__ import annotations

import argparse
import json

from rlm.cli.common import add_backend_arg, add_data_root_arg, add_profile_args
from rlm.core.services.diagnostics_service import DiagnosticsService


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="rlm doctor", description="Diagnose environment, dependencies, and runtime readiness.")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--json", action="store_true", help="Emit JSON report")
    p.add_argument("--strict", action="store_true", help="Exit non-zero when any required check fails")
    p.add_argument("--symbol", default=None, help="Validate data availability for symbol")
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    report = DiagnosticsService().run(
        verbose=args.verbose,
        data_root=args.data_root,
        backend=args.backend,
        profile=args.profile,
        config_path=args.config,
        symbol=args.symbol,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        status_icon = {True: "[OK]  ", False: "[FAIL]", None: "[SKIP]"}
        print("\nRLM Doctor Report")
        print("=" * 60)
        for check in report.checks:
            print(f"  {status_icon.get(check.passed, '[?]   ')}  {check.name}")
            if check.detail and (args.verbose or check.passed is False):
                print(f"         {check.detail}")

    if args.strict and not report.all_passed:
        raise SystemExit(1)
    if not args.strict and not report.all_passed and not args.json:
        raise SystemExit(1)
