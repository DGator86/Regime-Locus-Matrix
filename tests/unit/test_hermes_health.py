from __future__ import annotations

from pathlib import Path

from rlm.hermes_facts import health


def test_gather_health_report_honors_crew_services_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_services: list[list[str]] = []

    def fake_gather_report(root: Path, services: list[str]) -> health.HealthReport:
        captured_services.append(list(services))
        return health.HealthReport(timestamp="2020-01-01T00:00:00Z")

    monkeypatch.setenv("CREW_SERVICES", "rlm-master-telegram, rlm-telegram")
    monkeypatch.setattr(health, "_gather_report", fake_gather_report)
    monkeypatch.setattr(health, "_try_restart_inactive_services", lambda root, report, services: [])

    out = health.gather_health_report(tmp_path)

    assert captured_services == [["rlm-master-telegram", "rlm-telegram"]]
    assert out["remediation_log"] == []


def test_gather_health_report_explicit_services_override_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_services: list[list[str]] = []

    def fake_gather_report(root: Path, services: list[str]) -> health.HealthReport:
        captured_services.append(list(services))
        return health.HealthReport(timestamp="2020-01-01T00:00:00Z")

    monkeypatch.setenv("CREW_SERVICES", "rlm-master-telegram")
    monkeypatch.setattr(health, "_gather_report", fake_gather_report)
    monkeypatch.setattr(health, "_try_restart_inactive_services", lambda root, report, services: [])

    health.gather_health_report(tmp_path, services=["regime-locus-master"])

    assert captured_services == [["regime-locus-master"]]
