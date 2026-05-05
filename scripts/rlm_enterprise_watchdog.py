#!/usr/bin/env python3
"""
RLM enterprise host watchdog (systemd: often installed as /opt/enterprise/agents/scotty.py).

This is NOT the Hermes \"Scotty\" LLM — that runs inside regime-locus-crew via Hermes + skills.
This process only:
  - polls systemd + CPU/RAM/disk
  - optionally pulls JSON from rlm.hermes_facts.health (same facts crew Scotty uses)
  - sends terse Telegram alerts + lightweight Ollama commentary

Configure via /opt/enterprise/config/.env (see bootstrap.sh).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import requests
from dotenv import load_dotenv

# Default enterprise layout; override with ENTERPRISE_ENV path.
_DEFAULT_ENTERPRISE_ENV = Path("/opt/enterprise/config/.env")


def _load_env() -> None:
    p = Path(os.environ.get("ENTERPRISE_ENV", str(_DEFAULT_ENTERPRISE_ENV)))
    if p.is_file():
        load_dotenv(p)


_load_env()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
SCOTTY_MODEL = os.getenv("SCOTTY_MODEL", "llama3.2:3b")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
POLL_INTERVAL = int(os.getenv("SCOTTY_POLL_SECONDS", "60"))
LOG_PATH = os.getenv("SCOTTY_LOG", "/var/log/enterprise/scotty.log")
CPU_WARN = float(os.getenv("CPU_WARN_PCT", "80"))
RAM_WARN = float(os.getenv("RAM_WARN_PCT", "85"))
DISK_WARN = float(os.getenv("DISK_WARN_PCT", "75"))
WATCHED_SERVICES = [
    s.strip() for s in os.getenv("WATCHED_SERVICES", "ollama").split(",") if s.strip()
]

RLM_ROOT = Path(os.getenv("RLM_ROOT", "/opt/Regime-Locus-Matrix")).resolve()
RLM_HEALTH_PYTHON = (os.getenv("RLM_HEALTH_PYTHON") or "").strip()
SCOTTY_RLM_HEALTH = os.getenv("SCOTTY_RLM_HEALTH", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# systemd unit names in docs vs shipped examples often differ.
_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "rlm-telegram": (
        "rlm-telegram",
        "rlm-telegram-bot",
        "rlm-systems-control-telegram",
    ),
}

Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [rlm-watchdog] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("rlm-watchdog")


def resolve_unit(name: str) -> str:
    candidates = _UNIT_ALIASES.get(name, (name,))
    for c in candidates:
        try:
            r = subprocess.run(
                ["systemctl", "show", f"{c}.service", "--property=LoadState", "--value"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.stdout.strip() == "loaded":
                return c
        except Exception:
            continue
    return name


def unit_load_state(base: str) -> str:
    try:
        r = subprocess.run(
            ["systemctl", "show", f"{base}.service", "--property=LoadState", "--value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def optional_rlm_health_payload() -> dict[str, object] | None:
    """Same structured facts as Hermes crew Scotty — optional second python interpreter."""
    if not SCOTTY_RLM_HEALTH:
        return None
    py = RLM_HEALTH_PYTHON or ""
    if not py:
        log.warning("SCOTTY_RLM_HEALTH=1 but RLM_HEALTH_PYTHON unset — skipping health JSON")
        return None
    if not Path(py).exists():
        log.warning("RLM_HEALTH_PYTHON=%s missing — skipping health JSON", py)
        return None
    root = str(RLM_ROOT)
    if not RLM_ROOT.is_dir():
        log.warning("RLM_ROOT=%s not a directory — skipping health JSON", root)
        return None
    code = (
        "import json,sys,os;"
        "from pathlib import Path;"
        f"p=Path({root!r});"
        "os.environ.setdefault('RLM_ROOT', str(p));"
        "sys.path[:0]=[str(p/'src'), str(p)];"
        "from rlm.hermes_facts.health import gather_health_report;"
        "print(json.dumps(gather_health_report(p)))"
    )
    try:
        r = subprocess.run(
            [py, "-c", code],
            capture_output=True,
            text=True,
            timeout=45,
            cwd=root,
        )
        if r.returncode != 0:
            log.warning("gather_health_report failed: %s", (r.stderr or r.stdout)[:400])
            return None
        return json.loads(r.stdout.strip())
    except Exception as exc:
        log.warning("health subprocess error: %s", exc)
        return None


def send_telegram(msg: str, priority: str = "normal") -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    icons = {"normal": "⚙️", "warn": "⚠️", "critical": "🚨", "ok": "✅", "restart": "🔧"}
    icon = icons.get(priority, "⚙️")
    text = f"{icon} *RLM Watch* | {datetime.now().strftime('%H:%M:%S')}\n{msg}"
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log.error("Telegram error: %s", e)


def get_snapshot() -> dict[str, object]:
    cpu = psutil.cpu_percent(interval=2)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    services: dict[str, str] = {}
    for svc in WATCHED_SERVICES:
        try:
            resolved = resolve_unit(svc)
            r = subprocess.run(
                ["systemctl", "is-active", resolved],
                capture_output=True,
                text=True,
                timeout=5,
            )
            services[svc] = r.stdout.strip()
        except Exception:
            services[svc] = "unknown"
    snap: dict[str, object] = {
        "ts": datetime.now().isoformat(),
        "cpu_pct": cpu,
        "ram_pct": mem.percent,
        "ram_used_gb": round(mem.used / 1e9, 1),
        "ram_total_gb": round(mem.total / 1e9, 1),
        "disk_pct": disk.percent,
        "disk_free_gb": round(disk.free / 1e9, 1),
        "services": services,
        "load_avg": list(os.getloadavg()),
        "rlm_root": str(RLM_ROOT),
    }
    health = optional_rlm_health_payload()
    if health is not None:
        snap["rlm_health"] = {
            "overall_ok": health.get("overall_ok"),
            "session": health.get("session"),
            "stale_files": health.get("stale_files"),
            "remediation_log": (health.get("remediation_log") or [])[-5:],
        }
    return snap


def restart_service(svc: str) -> str:
    resolved = resolve_unit(svc)
    if unit_load_state(resolved) != "loaded":
        return (
            "SKIP restart `%s`: no loaded systemd unit "
            "(install a Telegram unit from deploy/*.service.example "
            "or set WATCHED_SERVICES to match `systemctl list-unit-files | grep -E 'rlm|regime'`)"
        ) % svc
    try:
        subprocess.run(["systemctl", "restart", resolved], check=True, timeout=30)
        if resolved == svc:
            return f"Restarted `{svc}` OK"
        return f"Restarted `{svc}` → `{resolved}` OK"
    except Exception as e:
        return f"FAILED restart `{resolved}` (watch `{svc}`): {e}"


def scotty_assess(snap: dict[str, object]) -> str:
    """Tiny local LLM layer — same engineering voice as Hermes data_monitor skill."""
    prompt = f"""You are the RLM host engineer (offline sibling of Hermes skill \"data_monitor\").

Regime Locus Matrix: Python quant stack — regime/HMM forecasts, universe_trade_plans.json, optional IBKR.

Facts JSON:
{json.dumps(snap, default=str)}

Rules:
- Overnight/weekend: inactive regime-locus-master is often NORMAL (timers only).
- Only ONE trading loop among regime-locus-master, rlm-master-telegram, rlm-master-trader should be active.
- One Telegram long-poll client per bot token.
- If \"rlm_health\" present, treat overall_ok=false as degraded unless explained by market hours.

Reply in 2 short sentences (plain text). No markdown."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": SCOTTY_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150},
            },
            timeout=60,
        )
        return str(r.json().get("response", "")).strip()
    except Exception as e:
        return f"LLM unavailable: {e}"


def main() -> None:
    log.info(
        "RLM host watchdog online — services=%s RLM_ROOT=%s health_json=%s",
        ", ".join(WATCHED_SERVICES),
        RLM_ROOT,
        SCOTTY_RLM_HEALTH,
    )
    send_telegram(
        "Host watchdog online.\n"
        f"Watching: `{', '.join(WATCHED_SERVICES)}`\n"
        f"RLM_ROOT=`{RLM_ROOT}`\n"
        f"Hermes crew Scotty/Spock/Kirk run separately in `regime-locus-crew`.\n"
        f"Thresholds: CPU>{CPU_WARN}% | RAM>{RAM_WARN}% | Disk>{DISK_WARN}%",
        "ok",
    )
    streak = 0
    while True:
        try:
            snap = get_snapshot()
            alerts: list[str] = []
            for svc, status in (snap["services"] or {}).items():
                if status != "active":
                    msg = restart_service(str(svc))
                    if msg.startswith("SKIP"):
                        log.warning("%s", msg)
                    else:
                        send_telegram(msg, "restart")
            cpu = float(snap["cpu_pct"])
            ram = float(snap["ram_pct"])
            disk = float(snap["disk_pct"])
            if cpu > CPU_WARN:
                alerts.append(f"CPU {cpu:.0f}%")
            if ram > RAM_WARN:
                alerts.append(f"RAM {ram:.0f}% ({snap['ram_used_gb']}/{snap['ram_total_gb']}GB)")
            if disk > DISK_WARN:
                alerts.append(f"Disk {disk:.0f}% ({snap['disk_free_gb']}GB free)")
            if alerts:
                streak += 1
                assessment = scotty_assess(snap)
                priority = "critical" if streak >= 3 else "warn"
                send_telegram(f"*Stress:* {' | '.join(alerts)}\n_{assessment}_", priority)
                log.warning("ALERT streak=%d: %s", streak, " | ".join(alerts))
            else:
                streak = 0
                log.info(
                    "OK — CPU:%.0f%% RAM:%.0f%% Disk:%.0f%%",
                    cpu,
                    ram,
                    disk,
                )
        except Exception as e:
            log.error("Loop error: %s", e)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
