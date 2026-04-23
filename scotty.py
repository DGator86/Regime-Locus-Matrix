#!/usr/bin/env python3
"""SCOTTY — System Health Agent for Regime Locus Matrix"""

import os, sys, json, time, subprocess, logging, requests, psutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/opt/enterprise/config/.env")

OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
SCOTTY_MODEL   = os.getenv("SCOTTY_MODEL", "llama3.2:3b")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
POLL_INTERVAL  = int(os.getenv("SCOTTY_POLL_SECONDS", "60"))
LOG_PATH       = os.getenv("SCOTTY_LOG", "/var/log/enterprise/scotty.log")
CPU_WARN       = float(os.getenv("CPU_WARN_PCT", "80"))
RAM_WARN       = float(os.getenv("RAM_WARN_PCT", "85"))
DISK_WARN      = float(os.getenv("DISK_WARN_PCT", "75"))
WATCHED_SERVICES = [s.strip() for s in os.getenv("WATCHED_SERVICES","ollama").split(",") if s.strip()]

Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCOTTY] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("scotty")

def send_telegram(msg, priority="normal"):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    icons = {"normal":"⚙️","warn":"⚠️","critical":"🚨","ok":"✅","restart":"🔧"}
    icon = icons.get(priority, "⚙️")
    text = f"{icon} *SCOTTY* | {datetime.now().strftime('%H:%M:%S')}\n{msg}"
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        log.error("Telegram error: %s", e)

def get_snapshot():
    cpu = psutil.cpu_percent(interval=2)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    services = {}
    for svc in WATCHED_SERVICES:
        try:
            r = subprocess.run(["systemctl","is-active",svc], capture_output=True, text=True, timeout=5)
            services[svc] = r.stdout.strip()
        except Exception:
            services[svc] = "unknown"
    return {
        "ts": datetime.now().isoformat(),
        "cpu_pct": cpu,
        "ram_pct": mem.percent,
        "ram_used_gb": round(mem.used/1e9,1),
        "ram_total_gb": round(mem.total/1e9,1),
        "disk_pct": disk.percent,
        "disk_free_gb": round(disk.free/1e9,1),
        "services": services,
        "load_avg": list(os.getloadavg()),
    }

def restart_service(svc):
    try:
        subprocess.run(["systemctl","restart",svc], check=True, timeout=30)
        return f"Restarted `{svc}` OK"
    except Exception as e:
        return f"FAILED restart `{svc}`: {e}"

def scotty_assess(snap):
    prompt = f"""You are Scotty, chief engineer of a trading system.
Assess this snapshot in 2 sentences max. Be direct. Only flag real problems.
CPU>{CPU_WARN}% | RAM>{RAM_WARN}% | Disk>{DISK_WARN}% = bad.
Snapshot: {json.dumps(snap)}"""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": SCOTTY_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150}
        }, timeout=180)
        return r.json().get("response","").strip()
    except Exception as e:
        return f"LLM unavailable: {e}"

def main():
    log.info("SCOTTY ONLINE — watching: %s", ", ".join(WATCHED_SERVICES))
    send_telegram(
        f"Scotty online on Regime Locus VPS.\nWatching: `{', '.join(WATCHED_SERVICES)}`\n"
        f"Thresholds: CPU>{CPU_WARN}% | RAM>{RAM_WARN}% | Disk>{DISK_WARN}%", "ok"
    )
    streak = 0
    while True:
        try:
            snap = get_snapshot()
            alerts = []
            for svc, status in snap["services"].items():
                if status != "active":
                    msg = restart_service(svc)
                    send_telegram(msg, "restart")
            if snap["cpu_pct"] > CPU_WARN:
                alerts.append(f"CPU {snap['cpu_pct']:.0f}%")
            if snap["ram_pct"] > RAM_WARN:
                alerts.append(f"RAM {snap['ram_pct']:.0f}% ({snap['ram_used_gb']}/{snap['ram_total_gb']}GB)")
            if snap["disk_pct"] > DISK_WARN:
                alerts.append(f"Disk {snap['disk_pct']:.0f}% ({snap['disk_free_gb']}GB free)")
            if alerts:
                streak += 1
                assessment = scotty_assess(snap)
                priority = "critical" if streak >= 3 else "warn"
                send_telegram(f"*Stress:* {' | '.join(alerts)}\n_{assessment}_", priority)
                log.warning("ALERT streak=%d: %s", streak, " | ".join(alerts))
            else:
                streak = 0
                log.info("OK — CPU:%.0f%% RAM:%.0f%% Disk:%.0f%%",
                         snap["cpu_pct"], snap["ram_pct"], snap["disk_pct"])
        except Exception as e:
            log.error("Loop error: %s", e)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
