from __future__ import annotations

import json
import os
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


def test_master_sibling_active_does_not_degrade_health(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        health,
        "_check_services",
        lambda root, services: [
            health.ServiceStatus(
                name="regime-locus-master",
                active=False,
                sub_state="dead",
                load_state="loaded",
            ),
            health.ServiceStatus(
                name="rlm-master-telegram",
                active=True,
                sub_state="running",
                load_state="loaded",
            ),
        ],
    )
    monkeypatch.setattr(health, "_check_disk", lambda root: [])
    monkeypatch.setattr(health, "_check_staleness", lambda root: [])
    monkeypatch.setattr(health, "_check_logs", lambda root, services: [])
    monkeypatch.setattr(health, "_run_doctor", lambda root: "")
    monkeypatch.setattr(health, "session_label", lambda: "rth")
    monkeypatch.setattr(health, "is_scanner_window_open", lambda: True)

    report = health._gather_report(tmp_path, ["regime-locus-master", "rlm-master-telegram"])

    assert report.overall_ok is True


def test_auto_restart_skips_inactive_master_when_sibling_is_active(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    report = health.HealthReport(
        timestamp="2020-01-01T00:00:00Z",
        services=[
            health.ServiceStatus(
                name="regime-locus-master",
                active=False,
                sub_state="dead",
                load_state="loaded",
            ),
            health.ServiceStatus(
                name="rlm-master-telegram",
                active=True,
                sub_state="running",
                load_state="loaded",
            ),
        ],
    )

    def fake_run(cmd: list[str], **kwargs) -> object:
        calls.append(cmd)
        raise AssertionError("systemctl restart should not be called")

    monkeypatch.setattr(health.shutil, "which", lambda name: "/bin/systemctl")
    monkeypatch.setattr(health.subprocess, "run", fake_run)

    actions = health._try_restart_inactive_services(
        tmp_path,
        report,
        ["regime-locus-master", "rlm-master-telegram"],
    )

    assert calls == []
    assert actions == ["[auto] skip restart regime-locus-master.service (active mutually-exclusive sibling)"]


def test_auto_restart_skips_inactive_trader_outside_scanner_window(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    report = health.HealthReport(
        timestamp="2020-01-01T00:00:00Z",
        services=[
            health.ServiceStatus(
                name="rlm-master-trader",
                active=False,
                sub_state="dead",
                load_state="loaded",
            ),
        ],
    )

    def fake_run(cmd: list[str], **kwargs) -> object:
        calls.append(cmd)
        raise AssertionError("systemctl restart should not be called")

    monkeypatch.setattr(health.shutil, "which", lambda name: "/bin/systemctl")
    monkeypatch.setattr(health.subprocess, "run", fake_run)
    monkeypatch.setattr(health, "is_scanner_window_open", lambda: False)

    actions = health._try_restart_inactive_services(
        tmp_path,
        report,
        ["rlm-master-trader"],
    )

    assert calls == []
    assert actions == []


def test_inactive_trader_outside_scanner_window_does_not_degrade_health(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        health,
        "_check_services",
        lambda root, services: [
            health.ServiceStatus(name="rlm-master-trader", active=False, sub_state="dead", load_state="loaded"),
        ],
    )
    monkeypatch.setattr(health, "_check_disk", lambda root: [])
    monkeypatch.setattr(health, "_check_staleness", lambda root: [])
    monkeypatch.setattr(health, "_check_logs", lambda root, services: [])
    monkeypatch.setattr(health, "_run_doctor", lambda root: "")
    monkeypatch.setattr(health, "session_label", lambda: "after_hours")
    monkeypatch.setattr(health, "is_scanner_window_open", lambda: False)

    report = health._gather_report(tmp_path, ["rlm-master-trader"])

    assert report.overall_ok is True


def test_staleness_ignores_old_trade_log_when_equity_book_only(tmp_path: Path, monkeypatch) -> None:
    """Open equities without options rows must not require fresh options monitor CSV mtime."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    eq_state = {"plan-a": {"status": "open", "symbol": "SPY"}}
    (processed / "equity_positions_state.json").write_text(json.dumps(eq_state), encoding="utf-8")
    tlog = processed / "trade_log.csv"
    # Header-only / tiny file: options collector skips → open_opts == 0
    tlog.write_text("timestamp_utc,plan_id,symbol\n", encoding="utf-8")
    old = os.path.getmtime(tlog) - 3 * 3600
    os.utime(tlog, (old, old))

    monkeypatch.setattr(health, "_STALE_HOURS", {"trade_log.csv": 0.1})
    stale = health._check_staleness(tmp_path)
    assert stale == []


def test_staleness_flags_old_trade_log_when_options_open(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    rows = (
        "timestamp_utc,plan_id,symbol,closed\n"
        + "\n".join(
            f"2020-01-01T00:00:0{i}Z,p{i},SPY,0" for i in range(12)
        )
        + "\n"
    )
    (processed / "trade_log.csv").write_text(rows, encoding="utf-8")
    tlog = processed / "trade_log.csv"
    old = os.path.getmtime(tlog) - 3 * 3600
    os.utime(tlog, (old, old))

    monkeypatch.setattr(health, "_STALE_HOURS", {"trade_log.csv": 0.1})
    stale = health._check_staleness(tmp_path)
    assert stale and any("options monitor log" in s for s in stale)


def test_staleness_ignores_trade_log_when_no_active_plans(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    plans = {"results": [{"symbol": "SPY", "status": "skipped"}]}
    (processed / "universe_trade_plans.json").write_text(json.dumps(plans), encoding="utf-8")
    tlog = processed / "trade_log.csv"
    tlog.write_text("timestamp_utc,plan_id,symbol\n", encoding="utf-8")

    old = os.path.getmtime(tlog) - 3 * 3600
    os.utime(tlog, (old, old))

    monkeypatch.setattr(health, "_STALE_HOURS", {"trade_log.csv": 0.1})
    stale = health._check_staleness(tmp_path)
    assert stale == []


def test_staleness_suppresses_plans_while_pipeline_lock_recent(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    plans = {"results": [{"symbol": "SPY", "status": "active", "plan_id": "p1"}]}
    plans_path = processed / "universe_trade_plans.json"
    plans_path.write_text(json.dumps(plans), encoding="utf-8")
    old = os.path.getmtime(plans_path) - 3 * 3600
    os.utime(plans_path, (old, old))
    (processed / ".universe_pipeline.lock").write_text("pid=99999\n", encoding="utf-8")

    monkeypatch.setattr(health, "_STALE_HOURS", {"universe_trade_plans.json": 0.1})
    monkeypatch.setattr(health, "universe_pipeline_lock_recent", lambda root: True)

    stale = health._check_staleness(tmp_path)
    assert stale == []


def test_gather_health_sets_pipeline_in_progress(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / ".universe_pipeline.lock").write_text("pid=1\n", encoding="utf-8")

    monkeypatch.setattr(health, "_check_services", lambda root, services: [])
    monkeypatch.setattr(health, "_check_disk", lambda root: [])
    monkeypatch.setattr(health, "_check_staleness", lambda root: [])
    monkeypatch.setattr(health, "_check_logs", lambda root, services: [])
    monkeypatch.setattr(health, "_run_doctor", lambda root: "")
    monkeypatch.setattr(health, "session_label", lambda: "rth")
    monkeypatch.setattr(health, "is_scanner_window_open", lambda: True)
    monkeypatch.setattr(health, "universe_pipeline_lock_recent", lambda root: True)

    report = health._gather_report(tmp_path, ["rlm-master-trader"])
    assert report.pipeline_in_progress is True


def test_health_degrades_on_recent_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        health,
        "_check_services",
        lambda root, services: [
            health.ServiceStatus(name="rlm-master-trader", active=True, sub_state="running", load_state="loaded"),
        ],
    )
    monkeypatch.setattr(health, "_check_disk", lambda root: [])
    monkeypatch.setattr(health, "_check_staleness", lambda root: [])
    monkeypatch.setattr(health, "_check_logs", lambda root, services: ["Massive HTTP 401 for /v3/snapshot/options/SPY"])
    monkeypatch.setattr(health, "_run_doctor", lambda root: "")
    monkeypatch.setattr(health, "session_label", lambda: "rth")
    monkeypatch.setattr(health, "is_scanner_window_open", lambda: True)

    report = health._gather_report(tmp_path, ["rlm-master-trader"])
    assert report.overall_ok is False
