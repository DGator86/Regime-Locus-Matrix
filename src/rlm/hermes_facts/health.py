"""Structured health facts (pipeline / host gather + optional auto-restart).

Environment: ``RLM_HEALTH_SKIP_JOURNAL=1`` skips ``journalctl`` (typical on Windows);
``RLM_HEALTH_SKIP_DOCTOR=1`` skips ``rlm doctor --strict``. ``scripts/rlm_health_check.py``
sets both by default on ``win32``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rlm.utils.market_hours import is_scanner_window_open, session_label

_DEFAULT_SERVICES = [
    "regime-locus-master",
    "rlm-master-trader",
    "regime-locus-crew",
    "rlm-control-center",
    "rlm-systems-control-telegram",
    "rlm-master-telegram",
    "rlm-telegram-bot",
]

_MUTUALLY_EXCLUSIVE_SERVICE_GROUPS = (
    frozenset(
        {
            "regime-locus-master",
            "rlm-master-telegram",
            "rlm-master-trader",
        }
    ),
)

_restart_last_mono: dict[str, float] = {}

_STALE_HOURS = {
    "universe_trade_plans.json": 0.1,
    "trade_log.csv": 0.1,
    "equity_positions_state.json": 0.1,
}


def _resolve_services(services: Optional[list[str]]) -> list[str]:
    if services is not None:
        return list(services)
    raw = os.environ.get("CREW_SERVICES", "")
    configured = [s.strip() for s in raw.split(",") if s.strip()]
    return configured or list(_DEFAULT_SERVICES)


@dataclass
class ServiceStatus:
    name: str
    active: bool
    sub_state: str
    load_state: str


@dataclass
class DiskUsage:
    path: str
    used_gb: float
    total_gb: float
    pct: float


@dataclass
class HealthReport:
    timestamp: str
    services: list[ServiceStatus] = field(default_factory=list)
    disk: list[DiskUsage] = field(default_factory=list)
    stale_files: list[str] = field(default_factory=list)
    recent_errors: list[str] = field(default_factory=list)
    doctor_output: str = ""
    session: str = "unknown"
    overall_ok: bool = True

    def to_text(self) -> str:
        lines = [f"[Health report @ {self.timestamp} | {self.session}]"]
        for s in self.services:
            icon = "✓" if s.active else "✗"
            lines.append(f"  {icon} {s.name}: {s.sub_state} ({s.load_state})")
        for d in self.disk:
            lines.append(f"  Disk {d.path}: {d.used_gb:.1f}/{d.total_gb:.1f} GB ({d.pct:.0f}%)")
        if self.stale_files:
            lines.append(f"  STALE artefacts: {', '.join(self.stale_files)}")
        if self.recent_errors:
            lines.append(f"  Recent log errors ({len(self.recent_errors)}):")
            for e in self.recent_errors[:5]:
                lines.append(f"    {e}")
        if self.doctor_output:
            lines.append(f"  rlm doctor: {self.doctor_output[:300]}")
        lines.append(f"  Overall: {'OK' if self.overall_ok else 'DEGRADED'}")
        return "\n".join(lines)


def _journal_services(services: list[str]) -> list[str]:
    if any("run_crew" in (a or "") for a in sys.argv):
        return [s for s in services if s != "regime-locus-crew"]
    return services


def _mutually_exclusive_group(name: str) -> frozenset[str]:
    for group in _MUTUALLY_EXCLUSIVE_SERVICE_GROUPS:
        if name in group:
            return group
    return frozenset()


def _has_active_mutually_exclusive_sibling(name: str, statuses: list[ServiceStatus]) -> bool:
    group = _mutually_exclusive_group(name)
    if not group:
        return False
    return any(s.name in group and s.name != name and s.active for s in statuses)


def _has_ambiguous_mutually_exclusive_restart(name: str, statuses: list[ServiceStatus]) -> bool:
    group = _mutually_exclusive_group(name)
    if not group:
        return False
    loaded_members = [s for s in statuses if s.name in group and s.load_state == "loaded"]
    return len(loaded_members) > 1


def _check_services(root: Path, services: list[str]) -> list[ServiceStatus]:
    results: list[ServiceStatus] = []
    for name in services:
        try:
            r = subprocess.run(
                [
                    "systemctl",
                    "show",
                    f"{name}.service",
                    "--property=ActiveState,SubState,LoadState",
                    "--no-pager",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            props: dict[str, str] = {}
            for line in r.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    props[k.strip()] = v.strip()
            active = props.get("ActiveState", "") == "active"
            results.append(
                ServiceStatus(
                    name=name,
                    active=active,
                    sub_state=props.get("SubState", "unknown"),
                    load_state=props.get("LoadState", "unknown"),
                )
            )
        except Exception as exc:
            results.append(
                ServiceStatus(
                    name=name,
                    active=False,
                    sub_state="error",
                    load_state=str(exc)[:60],
                )
            )
    return results


def _check_disk(root: Path) -> list[DiskUsage]:
    paths = [str(root), str(root / "data")]
    seen: set[str] = set()
    results: list[DiskUsage] = []
    for p in paths:
        real = str(Path(p).resolve())
        if real in seen or not Path(p).exists():
            continue
        seen.add(real)
        try:
            usage = shutil.disk_usage(p)
            total_gb = usage.total / 1e9
            used_gb = usage.used / 1e9
            results.append(
                DiskUsage(
                    path=p,
                    used_gb=used_gb,
                    total_gb=total_gb,
                    pct=100 * usage.used / usage.total,
                )
            )
        except Exception:
            pass
    return results


def _check_staleness(root: Path) -> list[str]:
    processed = root / "data" / "processed"
    stale: list[str] = []
    now = time.time()
    for fname, max_hours in _STALE_HOURS.items():
        fpath = processed / fname
        if not fpath.exists():
            continue
        age_hours = (now - fpath.stat().st_mtime) / 3600
        if age_hours > max_hours:
            stale.append(f"{fname} ({age_hours:.1f}h old)")
    return stale


def _check_logs(root: Path, services: list[str], lines: int = 100) -> list[str]:
    if (os.environ.get("RLM_HEALTH_SKIP_JOURNAL") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return []
    errors: list[str] = []
    for service in _journal_services(services):
        try:
            r = subprocess.run(
                [
                    "journalctl",
                    "-u",
                    f"{service}.service",
                    "-n",
                    str(lines),
                    "--no-pager",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in r.stdout.splitlines():
                low = line.lower()
                if any(kw in low for kw in ("error", "traceback", "exception", "critical", "failed")):
                    errors.append(line[-180:])
        except Exception:
            pass
    return errors[:20]


def _resolve_doctor_python(root: Path) -> str:
    override = (os.environ.get("RLM_DOCTOR_PYTHON") or "").strip()
    if override:
        return override
    candidates: list[Path] = [
        root / ".venv" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
    ]
    venv = (os.environ.get("VIRTUAL_ENV") or "").strip()
    if venv:
        vp = Path(venv)
        candidates.append(vp / "bin" / "python")
        candidates.append(vp / "Scripts" / "python.exe")
    for p in candidates:
        if p.exists():
            return str(p)
    if sys.executable:
        return sys.executable
    return "python3"


def _run_doctor(root: Path) -> str:
    if (os.environ.get("RLM_HEALTH_SKIP_DOCTOR") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return "(skipped — set RLM_HEALTH_SKIP_DOCTOR=0 to run `rlm doctor`)"
    try:
        python = _resolve_doctor_python(root)
        r = subprocess.run(
            [python, "-m", "rlm", "doctor", "--strict"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(root),
            env={**os.environ, "PYTHONPATH": str(root / "src")},
        )
        out = (r.stdout + r.stderr).strip()
        return out[:600] if out else "(no output)"
    except Exception as exc:
        return f"(doctor failed: {exc})"


def _gather_report(root: Path, services: list[str]) -> HealthReport:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = HealthReport(timestamp=ts)
    report.services = _check_services(root, services)
    report.disk = _check_disk(root)
    report.stale_files = _check_staleness(root)
    report.recent_errors = _check_logs(root, services)
    report.doctor_output = _run_doctor(root)
    report.session = session_label()
    scanner_open = is_scanner_window_open()
    service_issues = []
    for s in report.services:
        if s.load_state != "loaded":
            continue
        if not s.active:
            if _has_active_mutually_exclusive_sibling(s.name, report.services):
                continue
            if s.name == "regime-locus-master" and not scanner_open:
                continue
            service_issues.append(s.name)
    degraded = (
        bool(service_issues) or any(d.pct > 90 for d in report.disk) or (scanner_open and bool(report.stale_files))
    )
    report.overall_ok = not degraded
    return report


def _try_restart_inactive_services(root: Path, report: HealthReport, services: list[str]) -> list[str]:
    raw = (
        os.environ.get("RLM_HEALTH_AUTO_RESTART") or os.environ.get("SCOTTY_AUTO_RESTART") or "1"
    ).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return []
    if shutil.which("systemctl") is None:
        return []
    allow_crew = (
        os.environ.get("RLM_HEALTH_RESTART_ALLOW_CREW") or os.environ.get("SCOTTY_RESTART_ALLOW_CREW") or ""
    ).strip() in ("1", "true", "yes")
    skip_crew = any("run_crew" in (a or "") for a in sys.argv)
    try:
        cooldown = float(
            (os.environ.get("RLM_HEALTH_RESTART_COOLDOWN_SEC") or os.environ.get("SCOTTY_RESTART_COOLDOWN_SEC") or "180").strip()
        )
    except ValueError:
        cooldown = 180.0
    cooldown = max(30.0, cooldown)
    now = time.monotonic()
    actions: list[str] = []
    for s in report.services:
        if s.load_state != "loaded":
            continue
        if s.active:
            continue
        key = s.name
        if _has_active_mutually_exclusive_sibling(s.name, report.services):
            actions.append(f"[auto] skip restart {key}.service (active mutually-exclusive sibling)")
            continue
        if _has_ambiguous_mutually_exclusive_restart(s.name, report.services):
            actions.append(f"[auto] skip restart {key}.service (ambiguous mutually-exclusive master group)")
            continue
        if s.name == "regime-locus-master" and not is_scanner_window_open():
            continue
        if key == "regime-locus-crew" and skip_crew and not allow_crew:
            actions.append(f"[auto] skip restart {key}.service (would stop this crew process)")
            continue
        prev = _restart_last_mono.get(key, 0.0)
        if now - prev < cooldown:
            actions.append(f"[auto] skip restart {key}.service (cooldown {cooldown:.0f}s)")
            continue
        try:
            r = subprocess.run(
                ["systemctl", "restart", f"{key}.service"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            _restart_last_mono[key] = now
            tail = ((r.stderr or "") + (r.stdout or "")).strip()[:200]
            if r.returncode == 0:
                actions.append(f"[auto] systemctl restart {key}.service — ok")
            else:
                actions.append(f"[auto] systemctl restart {key}.service — failed: {tail or r.returncode}")
        except Exception as exc:
            actions.append(f"[auto] restart {key}.service error: {exc}")
    return actions


def _report_to_dict(report: HealthReport) -> dict[str, Any]:
    d = asdict(report)
    return d


def gather_health_report(root: Path, services: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Return a JSON-serialisable dict of host / pipeline health (no LLM).

    Runs optional systemd auto-restart once (when enabled via env), then re-gathers if any action was taken.
    """
    root = root.resolve()
    svc = _resolve_services(services)
    report = _gather_report(root, svc)
    remed = _try_restart_inactive_services(root, report, svc)
    if remed:
        report = _gather_report(root, svc)
    out = _report_to_dict(report)
    out["remediation_log"] = remed
    out["report_text"] = report.to_text()
    return out
