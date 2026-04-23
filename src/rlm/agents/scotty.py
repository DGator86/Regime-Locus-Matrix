"""
Scotty — system health guardian.

"She cannae take much more o' this, Captain!"

Monitors:
  - systemd service status for all RLM services
  - disk space (data lake, repo root)
  - recent journal errors
  - data freshness (stale artefacts)
  - rlm doctor output (when CLI available)

Returns a structured health report and an LLM-synthesised diagnosis with
recommended actions in plain English.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rlm.agents.base import LLMClient, LLMConfig, Message
from rlm.utils.market_hours import is_scanner_window_open, session_label

# -----------------------------------------------------------------------
# Scotty's character system prompt
# -----------------------------------------------------------------------
_SCOTTY_SYSTEM = """\
You are Scotty, the Chief Engineer of the trading system starship.
Your job is to keep the engines (services, data pipelines, storage) running perfectly.
You are practical, direct, and a little dramatic when things go wrong.

NIGHT WATCH / AFTER HOURS:
- When the Market State is 'after_hours' or 'weekend', the starship is in power-save mode.
- Many engines (services like regime-locus-master) will be powered down. This is NORMAL.
- Do not freak out about "everything being dark." If the crew and telegram bots are alive, the ship is safe.
- Only report real engineering failures (disk full, database corruption, or the crew service itself failing).

Speak in the first person. Keep responses concise — 3-10 bullet points maximum.
Focus on WHAT is broken and HOW to fix it. If everything is fine, say so briefly.
Never pad the response. Format output as plain text (no markdown headers).
"""

# Services to watch — names match the systemd unit files (without .service)
_DEFAULT_SERVICES = [
    "regime-locus-master",
    "regime-locus-crew",
    "rlm-control-center",
    "rlm-telegram",
    "rlm-master-telegram",
    "rlm-telegram-bot",
]

# Monotonic time of last systemctl restart per unit (rate limit)
_restart_last_mono: dict[str, float] = {}

# Artefact staleness thresholds
_STALE_HOURS = {
    "universe_trade_plans.json": 2.0,
    "trade_log.csv": 4.0,
    "equity_positions_state.json": 4.0,
}


@dataclass
class ServiceStatus:
    name: str
    active: bool
    sub_state: str  # "running", "dead", "failed", …
    load_state: str  # "loaded", "not-found", …


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
        lines = [f"[Scotty health report @ {self.timestamp} | {self.session}]"]
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


class ScottyAgent:
    def __init__(
        self,
        root: Path,
        llm: Optional[LLMClient] = None,
        services: Optional[list[str]] = None,
    ) -> None:
        self.root = root
        self.llm = llm or LLMClient()
        self.services = services or _DEFAULT_SERVICES
        self._last_report: Optional[HealthReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> tuple[HealthReport, str]:
        """Run all health checks and return (report, llm_diagnosis)."""
        report = self._gather()
        remed: list[str] = self._try_restart_inactive_services(report)
        if remed:
            report = self._gather()
        self._last_report = report
        extra = "\n".join(remed) if remed else ""
        diagnosis = self._diagnose(report, remediation_notes=extra)
        return report, diagnosis

    def is_degraded(self) -> bool:
        return self._last_report is not None and not self._last_report.overall_ok

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------

    def _gather(self) -> HealthReport:
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        report = HealthReport(timestamp=ts)

        report.services = self._check_services()
        report.disk = self._check_disk()
        report.stale_files = self._check_staleness()
        report.recent_errors = self._check_logs()
        report.doctor_output = self._run_doctor()
        report.session = session_label()

        scanner_open = is_scanner_window_open()
        
        # Determine degradation
        # We only care about regime-locus-master being dead if the scanner window is open.
        service_issues = []
        for s in report.services:
            if s.load_state != "loaded":
                continue
            if not s.active:
                if s.name == "regime-locus-master" and not scanner_open:
                    # After hours, we expect the master to be down.
                    continue
                service_issues.append(s.name)

        degraded = (
            bool(service_issues)
            or any(d.pct > 90 for d in report.disk)
            or (scanner_open and bool(report.stale_files))
        )
        report.overall_ok = not degraded
        return report

    def _check_services(self) -> list[ServiceStatus]:
        results: list[ServiceStatus] = []
        for name in self.services:
            try:
                r = subprocess.run(
                    ["systemctl", "show", f"{name}.service",
                     "--property=ActiveState,SubState,LoadState",
                     "--no-pager"],
                    capture_output=True, text=True, timeout=5,
                )
                props: dict[str, str] = {}
                for line in r.stdout.splitlines():
                    if "=" in line:
                        k, _, v = line.partition("=")
                        props[k.strip()] = v.strip()
                active = props.get("ActiveState", "") == "active"
                results.append(ServiceStatus(
                    name=name,
                    active=active,
                    sub_state=props.get("SubState", "unknown"),
                    load_state=props.get("LoadState", "unknown"),
                ))
            except Exception as exc:
                results.append(ServiceStatus(
                    name=name, active=False,
                    sub_state="error", load_state=str(exc)[:60],
                ))
        return results

    def _check_disk(self) -> list[DiskUsage]:
        paths = [str(self.root), str(self.root / "data")]
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
                results.append(DiskUsage(
                    path=p,
                    used_gb=used_gb,
                    total_gb=total_gb,
                    pct=100 * usage.used / usage.total,
                ))
            except Exception:
                pass
        return results

    def _check_staleness(self) -> list[str]:
        processed = self.root / "data" / "processed"
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

    def _journal_services(self) -> list[str]:
        """Units to scan with journalctl. Skip crew's own unit when this process is the crew."""
        if any("run_crew" in (a or "") for a in sys.argv):
            return [s for s in self.services if s != "regime-locus-crew"]
        return self.services

    def _check_logs(self, lines: int = 100) -> list[str]:
        errors: list[str] = []
        for service in self._journal_services():
            try:
                r = subprocess.run(
                    ["journalctl", "-u", f"{service}.service",
                     "-n", str(lines), "--no-pager", "-q"],
                    capture_output=True, text=True, timeout=10,
                )
                for line in r.stdout.splitlines():
                    low = line.lower()
                    if any(kw in low for kw in ("error", "traceback", "exception", "critical", "failed")):
                        errors.append(line[-180:])  # trim long lines
            except Exception:
                pass
        return errors[:20]

    def _resolve_doctor_python(self) -> str:
        override = (os.environ.get("RLM_DOCTOR_PYTHON") or "").strip()
        if override:
            return override
        candidates: list[Path] = [
            self.root / ".venv" / "bin" / "python",
            self.root / ".venv" / "Scripts" / "python.exe",
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

    def _run_doctor(self) -> str:
        try:
            python = self._resolve_doctor_python()
            r = subprocess.run(
                [python, "-m", "rlm", "doctor", "--strict"],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.root),
                env={**os.environ, "PYTHONPATH": str(self.root / "src")},
            )
            out = (r.stdout + r.stderr).strip()
            return out[:600] if out else "(no output)"
        except Exception as exc:
            return f"(doctor failed: {exc})"

    def _try_restart_inactive_services(self, report: HealthReport) -> list[str]:
        """``systemctl restart`` for loaded-but-inactive watched units (Linux VPS only)."""
        raw = (os.environ.get("SCOTTY_AUTO_RESTART") or "1").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return []
        if shutil.which("systemctl") is None:
            return []
        allow_crew = (os.environ.get("SCOTTY_RESTART_ALLOW_CREW") or "").strip() in (
            "1",
            "true",
            "yes",
        )
        skip_crew = any("run_crew" in (a or "") for a in sys.argv)
        try:
            cooldown = float((os.environ.get("SCOTTY_RESTART_COOLDOWN_SEC") or "180").strip())
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
            if s.name == "regime-locus-master" and not is_scanner_window_open():
                continue
            key = s.name
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

    # ------------------------------------------------------------------
    # LLM diagnosis
    # ------------------------------------------------------------------

    def _diagnose(self, report: HealthReport, *, remediation_notes: str = "") -> str:
        raw = report.to_text()
        if remediation_notes:
            raw = f"{raw}\n\nActions just taken (automated):\n{remediation_notes}"
        try:
            return self.llm.chat(
                [
                    Message(
                        "user",
                        "Here is the current system health report:\n\n"
                        f"{raw}\n\n"
                        "What is your diagnosis and what actions do I need to take right now?",
                    )
                ],
                system=_SCOTTY_SYSTEM,
            )
        except Exception as exc:
            return f"[Scotty LLM unavailable: {exc}]\n{raw}"
