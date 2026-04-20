"""DiagnosticsService — CI/ops oriented runtime checks."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlm.core.config import load_profile
from rlm.data.backend import DataBackend
from rlm.data.lake.readers import lake_has_bars
from rlm.data.paths import get_data_root, get_processed_data_dir


@dataclass
class CheckResult:
    name: str
    passed: bool | None
    detail: str = ""


@dataclass
class DiagnosticsReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed is not False for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"all_passed": self.all_passed, "checks": [asdict(c) for c in self.checks]}


class DiagnosticsService:
    def run(
        self,
        verbose: bool = False,
        data_root: str | None = None,
        backend: str = "auto",
        profile: str | None = None,
        config_path: str | None = None,
        symbol: str | None = None,
    ) -> DiagnosticsReport:
        report = DiagnosticsReport()
        report.checks += self._check_python()
        report.checks += self._check_package_importable()
        report.checks += self._check_core_deps()
        report.checks += self._check_filesystem(data_root)
        report.checks += self._check_profile(profile, config_path)
        report.checks += self._check_backend(backend, data_root, symbol)
        return report

    def _check_python(self) -> list[CheckResult]:
        v = sys.version_info
        return [CheckResult(name=f"Python {v.major}.{v.minor}.{v.micro}", passed=v >= (3, 10))]

    def _check_package_importable(self) -> list[CheckResult]:
        try:
            import rlm  # noqa: F401
            return [CheckResult(name="package: rlm importable", passed=True)]
        except ImportError as exc:
            return [CheckResult(name="package: rlm importable", passed=False, detail=str(exc))]

    def _check_core_deps(self) -> list[CheckResult]:
        required = ["numpy", "pandas", "pyyaml"]
        out = []
        for pkg in required:
            try:
                importlib.import_module(pkg)
                out.append(CheckResult(name=f"dep: {pkg}", passed=True))
            except ImportError as exc:
                out.append(CheckResult(name=f"dep: {pkg}", passed=False, detail=str(exc)))
        return out

    def _check_filesystem(self, data_root: str | None) -> list[CheckResult]:
        root = get_data_root(data_root)
        processed = get_processed_data_dir(data_root)
        artifacts = root / "artifacts"
        try:
            artifacts.mkdir(parents=True, exist_ok=True)
            writable_artifacts = os.access(artifacts, os.W_OK)
        except OSError:
            writable_artifacts = False
        return [
            CheckResult(name=f"fs: data root = {root}", passed=True),
            CheckResult(name="fs: processed writable", passed=os.access(processed, os.W_OK), detail=str(processed)),
            CheckResult(name="fs: artifacts writable", passed=writable_artifacts, detail=str(artifacts)),
        ]

    def _check_profile(self, profile: str | None, config_path: str | None) -> list[CheckResult]:
        try:
            if config_path:
                load_profile(path=config_path)
                return [CheckResult(name="config: explicit config valid", passed=True)]
            if profile:
                load_profile(name=profile)
                return [CheckResult(name=f"config: profile '{profile}' valid", passed=True)]
            return [CheckResult(name="config: profile/config", passed=True, detail="default runtime")]
        except Exception as exc:
            return [CheckResult(name="config: profile/config", passed=False, detail=str(exc))]

    def _check_backend(self, backend: str, data_root: str | None, symbol: str | None) -> list[CheckResult]:
        try:
            b = DataBackend.coerce(backend)
        except ValueError as exc:
            return [CheckResult(name="backend: value", passed=False, detail=str(exc))]

        checks = [CheckResult(name=f"backend: {b.value}", passed=True)]
        if symbol:
            sym = symbol.upper()
            raw_csv = get_data_root(data_root) / "raw" / f"bars_{sym}.csv"
            has_csv = raw_csv.is_file()
            has_lake = lake_has_bars(sym, data_root=data_root)
            ok = has_lake if b == DataBackend.LAKE else (has_csv if b == DataBackend.CSV else (has_lake or has_csv))
            checks.append(
                CheckResult(
                    name=f"symbol data: {sym}",
                    passed=ok,
                    detail=f"lake={has_lake} csv={has_csv}",
                )
            )
        return checks
