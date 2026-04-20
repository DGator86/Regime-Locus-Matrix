"""DiagnosticsService — environment, dependency, provider, and data lake health checks."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rlm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool | None  # None = skipped
    detail: str = ""


@dataclass
class DiagnosticsReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed is not False for c in self.checks)


class DiagnosticsService:
    """Run a battery of health checks and return a structured report."""

    def run(self, verbose: bool = False) -> DiagnosticsReport:
        report = DiagnosticsReport()

        report.checks += self._check_python()
        report.checks += self._check_core_deps()
        report.checks += self._check_optional_deps()
        report.checks += self._check_data_lake()
        report.checks += self._check_providers()

        return report

    # ------------------------------------------------------------------
    # Python / environment
    # ------------------------------------------------------------------

    def _check_python(self) -> list[CheckResult]:
        version = sys.version_info
        ok = version >= (3, 10)
        return [CheckResult(
            name=f"Python {version.major}.{version.minor}",
            passed=ok,
            detail="" if ok else "Python 3.10+ required",
        )]

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------

    def _check_core_deps(self) -> list[CheckResult]:
        required = ["numpy", "pandas", "scipy", "statsmodels", "hmmlearn", "optuna", "pydantic"]
        results = []
        for pkg in required:
            try:
                importlib.import_module(pkg)
                results.append(CheckResult(name=f"dep: {pkg}", passed=True))
            except ImportError as exc:
                results.append(CheckResult(name=f"dep: {pkg}", passed=False, detail=str(exc)))
        return results

    def _check_optional_deps(self) -> list[CheckResult]:
        optional = {
            "torch": "kronos",
            "duckdb": "microstructure",
            "streamlit": "ui",
            "ibapi": "ibkr",
            "boto3": "flatfiles",
        }
        results = []
        for pkg, extra in optional.items():
            try:
                importlib.import_module(pkg)
                results.append(CheckResult(name=f"opt: {pkg} [{extra}]", passed=True))
            except ImportError:
                results.append(CheckResult(
                    name=f"opt: {pkg} [{extra}]",
                    passed=None,
                    detail=f"Not installed. pip install -e '.[{extra}]'",
                ))
        return results

    # ------------------------------------------------------------------
    # Data lake paths
    # ------------------------------------------------------------------

    def _check_data_lake(self) -> list[CheckResult]:
        root = Path(__file__).resolve().parents[5]
        checks = [
            ("data/raw", "Raw bars / chains directory"),
            ("data/processed", "Processed outputs directory"),
        ]
        results = []
        for rel, label in checks:
            path = root / rel
            results.append(CheckResult(
                name=f"lake: {rel}",
                passed=path.is_dir(),
                detail="" if path.is_dir() else f"Missing: {path}",
            ))
        return results

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def _check_providers(self) -> list[CheckResult]:
        results: list[CheckResult] = []

        # yfinance connectivity (lightweight)
        try:
            import yfinance as yf
            ticker = yf.Ticker("SPY")
            info = ticker.fast_info
            _ = info.last_price  # triggers a network call
            results.append(CheckResult(name="provider: yfinance", passed=True))
        except Exception as exc:
            results.append(CheckResult(name="provider: yfinance", passed=False, detail=str(exc)))

        # IBKR — check ibapi import only; connection check would require running TWS
        try:
            import ibapi  # noqa: F401
            results.append(CheckResult(
                name="provider: ibkr (import)",
                passed=True,
                detail="ibapi installed; connection requires running TWS/Gateway",
            ))
        except ImportError:
            results.append(CheckResult(
                name="provider: ibkr (import)",
                passed=None,
                detail="ibapi not installed — pip install ibapi",
            ))

        return results
