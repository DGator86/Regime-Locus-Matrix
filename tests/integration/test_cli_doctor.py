"""Integration tests for DiagnosticsService and the doctor command."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from rlm.core.services.diagnostics_service import DiagnosticsReport, DiagnosticsService


class TestDiagnosticsServiceRun:
    def test_run_returns_report(self, tmp_path: Path) -> None:
        svc = DiagnosticsService()
        report = svc.run(data_root=str(tmp_path))
        assert isinstance(report, DiagnosticsReport)

    def test_report_has_checks(self, tmp_path: Path) -> None:
        report = DiagnosticsService().run(data_root=str(tmp_path))
        assert len(report.checks) > 0

    def test_python_version_check_passes(self, tmp_path: Path) -> None:
        report = DiagnosticsService().run(data_root=str(tmp_path))
        python_checks = [c for c in report.checks if "Python" in c.name]
        assert len(python_checks) == 1
        # Must pass on any supported Python (3.10+)
        assert python_checks[0].passed is True

    def test_package_importable_check_passes(self, tmp_path: Path) -> None:
        report = DiagnosticsService().run(data_root=str(tmp_path))
        import_checks = [c for c in report.checks if "rlm importable" in c.name]
        assert len(import_checks) == 1
        assert import_checks[0].passed is True

    def test_all_passed_property_true_when_no_failures(self, tmp_path: Path) -> None:
        report = DiagnosticsService().run(data_root=str(tmp_path), backend="csv")
        # all_passed should be True when no check has passed=False
        hard_fails = [c for c in report.checks if c.passed is False]
        if not hard_fails:
            assert report.all_passed is True

    def test_all_passed_false_when_failure_present(self) -> None:
        from rlm.core.services.diagnostics_service import CheckResult
        report = DiagnosticsReport(checks=[
            CheckResult(name="ok", passed=True),
            CheckResult(name="bad", passed=False, detail="fail"),
        ])
        assert report.all_passed is False

    def test_all_passed_true_with_skips(self) -> None:
        from rlm.core.services.diagnostics_service import CheckResult
        report = DiagnosticsReport(checks=[
            CheckResult(name="ok", passed=True),
            CheckResult(name="skip", passed=None, detail="optional"),
        ])
        assert report.all_passed is True

    def test_backend_checks_included_for_auto(self, tmp_path: Path) -> None:
        report = DiagnosticsService().run(data_root=str(tmp_path), backend="auto")
        backend_checks = [c for c in report.checks if "backend" in c.name.lower()]
        assert len(backend_checks) > 0

    def test_symbol_data_checks_included_when_symbol_given(self, tmp_path: Path) -> None:
        report = DiagnosticsService().run(data_root=str(tmp_path), symbol="SPY")
        symbol_checks = [c for c in report.checks if "SPY" in c.name]
        assert len(symbol_checks) > 0

    def test_symbol_bars_check_fails_when_file_missing(self, tmp_path: Path) -> None:
        report = DiagnosticsService().run(data_root=str(tmp_path), symbol="SPY")
        bars_checks = [c for c in report.checks if "SPY" in c.name and "bars" in c.name.lower()]
        assert len(bars_checks) >= 1
        assert bars_checks[0].passed is False

    def test_profile_check_passes_with_no_profile(self, tmp_path: Path) -> None:
        report = DiagnosticsService().run(data_root=str(tmp_path), profile=None, config_path=None)
        profile_checks = [c for c in report.checks if "profile" in c.name.lower()]
        assert len(profile_checks) >= 1
        assert profile_checks[0].passed is True


class TestDoctorCLI:
    def _run_doctor_json(self, tmp_path: Path) -> dict:
        from rlm.cli.doctor import _print_json
        report = DiagnosticsService().run(data_root=str(tmp_path), backend="csv")
        buf = StringIO()
        with patch("sys.stdout", buf):
            _print_json(report)
        return json.loads(buf.getvalue())

    def test_json_output_has_all_passed_key(self, tmp_path: Path) -> None:
        data = self._run_doctor_json(tmp_path)
        assert "all_passed" in data

    def test_json_output_has_checks_list(self, tmp_path: Path) -> None:
        data = self._run_doctor_json(tmp_path)
        assert "checks" in data
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) > 0

    def test_json_checks_have_required_fields(self, tmp_path: Path) -> None:
        data = self._run_doctor_json(tmp_path)
        for check in data["checks"]:
            assert "name" in check
            assert "passed" in check
            assert "detail" in check

    def test_json_summary_has_counts(self, tmp_path: Path) -> None:
        data = self._run_doctor_json(tmp_path)
        assert "summary" in data
        s = data["summary"]
        assert "passed" in s
        assert "skipped" in s
        assert "failed" in s
        assert "total" in s
        assert s["total"] == len(data["checks"])
