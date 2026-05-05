#!/bin/bash
# =============================================================================
#  REGIME LOCUS MATRIX — VPS Bootstrap
#  Fresh Ubuntu 24.04 → Fully running system
# =============================================================================

set -euo pipefail

# ── Credentials (pre-filled) ─────────────────────────────────────────────────
TELEGRAM_TOKEN="8659637634:AAEH_LhTF8zW8Z-X8O0xoU44CAA2R_10JOM"
GITHUB_REPO="https://github.com/DGator86/Regime-Locus-Matrix"
BOT_USERNAME="@rlm_options_alert_bot"
CHAT_ID="8236163940" # Hardcoded to bypass interactive prompt

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="/opt/regime-locus"
ENTERPRISE_DIR="/opt/enterprise"
LOG_DIR="/var/log/enterprise"
VENV="$REPO_DIR/venv"

# ── Colors ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; A='\033[0;33m'; B='\033[0;34m'; R='\033[0;31m'; NC='\033[0m'
log()   { echo -e "${G}[✓]${NC} $1"; }
info()  { echo -e "${B}[→]${NC} $1"; }
warn()  { echo -e "${A}[!]${NC} $1"; }
error() { echo -e "${R}[✗]${NC} $1"; exit 1; }
step()  { echo -e "\n${B}━━━ $1 ━━━${NC}"; }

# ── Preflight ────────────────────────────────────────────────────────────────
[[ $EUID -ne 0 ]] && error "Run as root: sudo bash bootstrap.sh"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   REGIME LOCUS MATRIX — Full VPS Bootstrap           ║"
echo "║   Scotty  •  Spock  •  Kirk  •  Regime Locus Bot    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
DISK_GB=$(df -BG / | awk 'NR==2{print $4}' | tr -d 'G')
CORES=$(nproc)
echo "  System: ${RAM_GB}GB RAM | ${CORES} cores | ${DISK_GB}GB free"
[[ $RAM_GB -lt 7 ]]  && error "Need 8GB+ RAM"
[[ $DISK_GB -lt 14 ]] && error "Need 15GB+ free disk"
log "System OK"


# ══════════════════════════════════════════════════════════════════════════════
step "1 / 10  System packages"
# ══════════════════════════════════════════════════════════════════════════════
apt-get update -qq
apt-get install -y -qq \
    git curl wget jq htop \
    python3 python3-pip python3-venv python3-dev \
    build-essential libssl-dev \
    systemd
log "System packages installed"


# ══════════════════════════════════════════════════════════════════════════════
step "2 / 10  Clone Regime-Locus-Matrix"
# ══════════════════════════════════════════════════════════════════════════════
if [[ -d "$REPO_DIR/.git" ]]; then
    warn "Repo already exists — pulling latest"
    git -C "$REPO_DIR" pull --ff-only
else
    git clone "$GITHUB_REPO" "$REPO_DIR"
fi
log "Repo at $REPO_DIR"


# ══════════════════════════════════════════════════════════════════════════════
step "3 / 10  Python virtualenv + repo dependencies"
# ══════════════════════════════════════════════════════════════════════════════
python3 -m venv "$VENV"
source "$VENV/bin/activate"

# Core deps always needed
pip install --upgrade pip -q
pip install requests psutil python-dotenv rich schedule -q

# Install repo requirements if they exist
for REQ in "$REPO_DIR/requirements.txt" "$REPO_DIR/requirements-prod.txt"; do
    if [[ -f "$REQ" ]]; then
        info "Installing from $REQ..."
        pip install -r "$REQ" -q && log "Installed $REQ"
    fi
done

deactivate
log "Virtual environment ready at $VENV"


# ══════════════════════════════════════════════════════════════════════════════
step "4 / 10  Install Ollama"
# ══════════════════════════════════════════════════════════════════════════════
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    warn "Ollama already installed — skipping"
fi

cat > /etc/systemd/system/ollama.service << 'OLLAMASVC'
[Unit]
Description=Ollama LLM Inference Server
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=root
Restart=always
RestartSec=5
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_KEEP_ALIVE=15m"
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
OLLAMASVC

systemctl daemon-reload
systemctl enable --now ollama

info "Waiting for Ollama API..."
for i in {1..40}; do
    curl -s http://localhost:11434/api/tags &>/dev/null && break || sleep 3
    [[ $i -eq 40 ]] && error "Ollama failed to start"
done
log "Ollama running on :11434"


# ══════════════════════════════════════════════════════════════════════════════
step "5 / 10  Pull AI models"
# ══════════════════════════════════════════════════════════════════════════════
info "Pulling Scotty model: llama3.2:3b (~2.0 GB — Scotty brain)..."
ollama pull llama3.2:3b && log "llama3.2:3b ready"

info "Pulling Spock model: qwen2.5:7b (~4.7 GB — Spock brain)..."
ollama pull qwen2.5:7b && log "qwen2.5:7b ready"

ollama list


# ══════════════════════════════════════════════════════════════════════════════
step "6 / 10  Auto-detect Telegram Chat ID (Bypassed)"
# ══════════════════════════════════════════════════════════════════════════════
info "Bypassed interactive prompt, using hardcoded CHAT_ID=$CHAT_ID"


# ══════════════════════════════════════════════════════════════════════════════
step "7 / 10  Write configuration"
# ══════════════════════════════════════════════════════════════════════════════
mkdir -p "$ENTERPRISE_DIR"/{agents,config,data}
mkdir -p "$LOG_DIR"

cat > "$ENTERPRISE_DIR/config/.env" << ENV
# ═══════════════════════════════════════════════════
#  REGIME LOCUS MATRIX — Enterprise Config
#  Generated: $(date)
# ═══════════════════════════════════════════════════

# ── Repo ────────────────────────────────────────────
REPO_DIR=${REPO_DIR}
VENV=${VENV}

# ── Ollama ──────────────────────────────────────────
OLLAMA_URL=http://localhost:11434/api/generate
SCOTTY_MODEL=llama3.2:3b
SPOCK_MODEL=qwen2.5:7b

# ── Telegram / Regime Locus Bot ─────────────────────
TELEGRAM_BOT_TOKEN=${TELEGRAM_TOKEN}
TELEGRAM_CHAT_ID=${CHAT_ID}

# ── Scotty ──────────────────────────────────────────
SCOTTY_POLL_SECONDS=60
SCOTTY_LOG=${LOG_DIR}/scotty.log
CPU_WARN_PCT=80
RAM_WARN_PCT=85
DISK_WARN_PCT=75

# Services Scotty watches (add your app services here after launch)
WATCHED_SERVICES=ollama,regime-locus

# ── Spock ───────────────────────────────────────────
SPOCK_TIMEOUT_SECONDS=90
SPOCK_MIN_CONFIDENCE=0.65
SPOCK_ENABLED=true
SPOCK_OVERRIDE_KEY=

# ── Kirk ────────────────────────────────────────────
DECISION_LOG=${ENTERPRISE_DIR}/data/decisions.jsonl
ENV

chmod 600 "$ENTERPRISE_DIR/config/.env"
log "Config written to $ENTERPRISE_DIR/config/.env"


# ══════════════════════════════════════════════════════════════════════════════
step "8 / 10  Install Enterprise agents"
# ══════════════════════════════════════════════════════════════════════════════

# Write scotty.py inline
cat > "$ENTERPRISE_DIR/agents/scotty.py" << 'SCOTTY_PY'
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

# systemd unit names in docs vs shipped examples often differ (rlm-telegram vs rlm-telegram-bot).
_UNIT_ALIASES = {
    "rlm-telegram": ("rlm-telegram", "rlm-telegram-bot", "rlm-systems-control-telegram"),
}


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
            resolved = resolve_unit(svc)
            r = subprocess.run(["systemctl", "is-active", resolved], capture_output=True, text=True, timeout=5)
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
    resolved = resolve_unit(svc)
    if unit_load_state(resolved) != "loaded":
        return (
            "SKIP restart `%s`: no loaded systemd unit "
            "(install deploy/rlm-telegram.service.example as /etc/systemd/system/rlm-telegram.service "
            "or rename WATCHED_SERVICES to match `systemctl list-unit-files | grep telegram`)"
        ) % svc
    try:
        subprocess.run(["systemctl", "restart", resolved], check=True, timeout=30)
        if resolved == svc:
            return f"Restarted `{svc}` OK"
        return f"Restarted `{svc}` → `{resolved}` OK"
    except Exception as e:
        return f"FAILED restart `{resolved}` (watch `{svc}`): {e}"

def scotty_assess(snap):
    prompt = f"""You are Scotty, chief engineer of a trading system.
Assess this snapshot in 2 sentences max. Be direct. Only flag real problems.
CPU>{CPU_WARN}% | RAM>{RAM_WARN}% | Disk>{DISK_WARN}% = bad.
Snapshot: {json.dumps(snap)}"""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": SCOTTY_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150}
        }, timeout=60)
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
                    if msg.startswith("SKIP"):
                        log.warning("%s", msg)
                    else:
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
SCOTTY_PY

# Write spock.py inline
cat > "$ENTERPRISE_DIR/agents/spock.py" << 'SPOCK_PY'
#!/usr/bin/env python3
"""SPOCK — Analytical Advisor for Regime Locus Matrix"""

import os, sys, json, re, logging, requests, argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("/opt/enterprise/config/.env")

OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
SPOCK_MODEL    = os.getenv("SPOCK_MODEL", "qwen2.5:7b")
SPOCK_TIMEOUT  = int(os.getenv("SPOCK_TIMEOUT_SECONDS", "90"))
MIN_CONF       = float(os.getenv("SPOCK_MIN_CONFIDENCE", "0.65"))

log = logging.getLogger("spock")

SPOCK_SYSTEM = """You are Spock — science officer and logical advisor to a trading system.
Pure data-driven analysis only. No speculation. No emotion.

Respond ONLY in valid JSON. No markdown, no prose, no preamble.

Format:
{
  "assessment": "one sentence probability statement",
  "factors": ["factor 1", "factor 2", "factor 3"],
  "action": "BUY | SELL | HOLD | ABORT",
  "confidence": 0.00,
  "risk": "primary risk in one sentence"
}

Rules:
- confidence < 0.5 → ABORT or HOLD
- Never recommend action when data insufficient
- If gnosis_projection confidence < 0.6, reduce your confidence
- If open_positions >= 5, prefer HOLD
"""

def spock_analyze(trade_context: dict) -> dict:
    prompt = f"Analyze this trade:\n{json.dumps(trade_context, indent=2, default=str)}"
    t0 = datetime.now()
    raw = ""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": SPOCK_MODEL,
            "system": SPOCK_SYSTEM,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.05, "num_predict": 400}
        }, timeout=SPOCK_TIMEOUT)
        raw = r.json().get("response", "").strip()
    except Exception as e:
        log.error("Spock error: %s", e)
        return {"proceed": False, "action": "ABORT", "confidence": 0.0,
                "assessment": f"LLM error: {e}", "factors": [], "risk": "System error", "raw": ""}

    elapsed = int((datetime.now() - t0).total_seconds() * 1000)
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"proceed": False, "action": "ABORT", "confidence": 0.0,
                "assessment": "Parse failed", "factors": [], "risk": "Unparseable response",
                "raw": raw, "elapsed_ms": elapsed}
    try:
        data = json.loads(match.group())
        confidence = float(data.get("confidence", 0.0))
        action = str(data.get("action","ABORT")).upper()
        proceed = action in ("BUY","SELL","LONG","SHORT") and confidence >= MIN_CONF
        return {
            "proceed": proceed, "action": action, "confidence": confidence,
            "assessment": data.get("assessment",""), "factors": data.get("factors",[]),
            "risk": data.get("risk",""), "raw": raw, "elapsed_ms": elapsed
        }
    except Exception as e:
        return {"proceed": False, "action": "ABORT", "confidence": 0.0,
                "assessment": f"Parse error: {e}", "factors": [], "risk": "Parse failure",
                "raw": raw, "elapsed_ms": elapsed}

def run_test():
    ctx = {
        "strategy": "regime_locus_test",
        "market": "options",
        "direction": "BUY",
        "current_price": 2.45,
        "position_size_usd": 300,
        "open_positions": 1,
        "recent_pnl_7d": 120,
        "market_state": "trending_bullish",
        "additional_context": "IV rank 34, delta 0.38, 21 DTE"
    }
    print("\n🖖 SPOCK TEST\n" + "─"*50)
    print(json.dumps(ctx, indent=2))
    print("\n⏳ Querying Spock (15-45s on CPU)...\n")
    v = spock_analyze(ctx)
    print("─"*50)
    print(f"ACTION:     {v['action']}")
    print(f"CONFIDENCE: {v['confidence']:.0%}")
    print(f"PROCEED:    {'YES ✅' if v['proceed'] else 'NO ❌'}")
    print(f"ASSESSMENT: {v['assessment']}")
    print(f"RISK:       {v['risk']}")
    print(f"ELAPSED:    {v.get('elapsed_ms',0)}ms")
    for f in v.get("factors",[]):
        print(f"  • {f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [SPOCK] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        run_test()
SPOCK_PY

# Kirk hook
cat > "$ENTERPRISE_DIR/agents/kirk_hook.py" << 'KIRK_PY'
#!/usr/bin/env python3
"""KIRK HOOK — Pre-trade Spock consultation layer"""

import os, sys, json, logging, time
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/opt/enterprise/config/.env")
sys.path.insert(0, str(Path(__file__).parent))
from spock import spock_analyze

log = logging.getLogger("kirk")
SPOCK_ENABLED   = os.getenv("SPOCK_ENABLED","true").lower() == "true"
MIN_CONFIDENCE  = float(os.getenv("SPOCK_MIN_CONFIDENCE","0.65"))
OVERRIDE_KEY    = os.getenv("SPOCK_OVERRIDE_KEY","")
DECISION_LOG    = os.getenv("DECISION_LOG","/opt/enterprise/data/decisions.jsonl")

@dataclass
class KirkDecision:
    proceed: bool
    action: str
    confidence: float
    reason: str
    risk: str
    factors: list = field(default_factory=list)
    elapsed_ms: int = 0
    bypassed: bool = False

def consult_spock(trade_context: dict, override: str = "") -> KirkDecision:
    if override and override == OVERRIDE_KEY:
        return KirkDecision(True, "MANUAL_OVERRIDE", 1.0, "Override", "None", bypassed=True)
    if not SPOCK_ENABLED:
        return KirkDecision(True, "UNANALYZED", 0.5, "Spock disabled", "None", bypassed=True)
    v = spock_analyze(trade_context)
    d = KirkDecision(
        proceed=v["proceed"], action=v["action"], confidence=v["confidence"],
        reason=v["assessment"], risk=v["risk"], factors=v.get("factors",[]),
        elapsed_ms=v.get("elapsed_ms",0)
    )
    _log_decision(trade_context, d)
    if d.proceed:
        log.info("✅ SPOCK APPROVES %s @ %.0f%% (%dms)", d.action, d.confidence*100, d.elapsed_ms)
    else:
        log.info("❌ SPOCK VETOES %s @ %.0f%% — %s", d.action, d.confidence*100, d.risk)
    return d

def _log_decision(ctx, d):
    try:
        Path(DECISION_LOG).parent.mkdir(parents=True, exist_ok=True)
        with open(DECISION_LOG,"a") as f:
            f.write(json.dumps({"ts":time.time(),"context":ctx,"decision":vars(d)})+"\n")
    except Exception as e:
        log.error("Decision log error: %s", e)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [KIRK] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    if args.stats:
        try:
            lines = open(DECISION_LOG).readlines()
            total = len(lines)
            approved = sum(1 for l in lines if '"proceed": true' in l)
            print(f"\nDecisions: {total} | Approved: {approved} | Vetoed: {total-approved}")
        except FileNotFoundError:
            print("No decisions logged yet.")
KIRK_PY

chmod +x "$ENTERPRISE_DIR/agents/"*.py
log "Enterprise agents installed"


# ══════════════════════════════════════════════════════════════════════════════
step "9 / 10  Register systemd services"
# ══════════════════════════════════════════════════════════════════════════════

# Scotty service
cat > /etc/systemd/system/scotty.service << SCOSERVICE
[Unit]
Description=Scotty — Regime Locus System Health Agent
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
ExecStart=${VENV}/bin/python3 ${ENTERPRISE_DIR}/agents/scotty.py
WorkingDirectory=${ENTERPRISE_DIR}
Restart=always
RestartSec=15
StandardOutput=append:${LOG_DIR}/scotty.log
StandardError=append:${LOG_DIR}/scotty_error.log
EnvironmentFile=${ENTERPRISE_DIR}/config/.env
MemoryMax=512M

[Install]
WantedBy=multi-user.target
SCOSERVICE

systemctl daemon-reload
systemctl enable scotty
systemctl start scotty
log "Scotty service started"


# ══════════════════════════════════════════════════════════════════════════════
step "10 / 10  Health check"
# ══════════════════════════════════════════════════════════════════════════════
sleep 5

echo ""
echo "  Service status:"
systemctl is-active --quiet ollama && echo "  ✅ ollama" || echo "  ❌ ollama"
systemctl is-active --quiet scotty && echo "  ✅ scotty" || echo "  ❌ scotty"

echo ""
echo "  Ollama models:"
ollama list

echo ""
echo "  Scotty log (last 5 lines):"
tail -5 "$LOG_DIR/scotty.log" 2>/dev/null || echo "  (no log yet)"


# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║             BOOTSTRAP COMPLETE                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Repo:     $REPO_DIR"
echo "  Agents:   $ENTERPRISE_DIR/agents/"
echo "  Config:   $ENTERPRISE_DIR/config/.env"
echo "  Logs:     $LOG_DIR/"
echo ""
echo "  Test Spock:"
echo "    $VENV/bin/python3 $ENTERPRISE_DIR/agents/spock.py --test"
echo ""
echo "  Live logs:"
echo "    tail -f $LOG_DIR/scotty.log"
echo ""
echo "  ⚠️  Regenerate your Telegram token at @BotFather!"
echo "     Then update: $ENTERPRISE_DIR/config/.env"
echo ""
