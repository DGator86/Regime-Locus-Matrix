#!/bin/bash
# =============================================================================
#  REGIME LOCUS MATRIX — VPS Bootstrap
#  Fresh Ubuntu 24.04 → Fully running system
# =============================================================================

set -euo pipefail

# ── Credentials (export before running — do not commit secrets) ──────────────
[[ -n "${TELEGRAM_TOKEN:-}" ]] || error "Export TELEGRAM_TOKEN from BotFather before bootstrap, e.g. sudo -E bash bootstrap.sh"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"
[[ -n "$TELEGRAM_CHAT_ID" ]] || warn "TELEGRAM_CHAT_ID unset — set later in /opt/enterprise/config/.env"

GITHUB_REPO="${GITHUB_REPO:-https://github.com/DGator86/Regime-Locus-Matrix}"

# ── Paths (RLM repo should match systemd WorkingDirectory on your VPS) ───────
REPO_DIR="${REPO_DIR:-/opt/Regime-Locus-Matrix}"
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
echo "║   RLM Watch • Hermes crew • Regime Locus Matrix                     ║"
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
step "6 / 10  Telegram Chat ID"
# ══════════════════════════════════════════════════════════════════════════════
info "Using TELEGRAM_CHAT_ID from environment (may be empty)."


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
TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}

# ── RLM Watch (host watchdog — NOT Hermes crew; crew runs regime-locus-crew) ──
RLM_ROOT=${REPO_DIR}
RLM_HEALTH_PYTHON=/opt/rlm-venv/bin/python
SCOTTY_RLM_HEALTH=0

# ── Scotty (legacy env prefix — process is RLM host watchdog) ─────────────────
SCOTTY_POLL_SECONDS=60
SCOTTY_LOG=${LOG_DIR}/scotty.log
CPU_WARN_PCT=80
RAM_WARN_PCT=85
DISK_WARN_PCT=75

# systemd base names — must match loaded units (see deploy/*.service.example).
# Typical Hostinger: telegram bot = rlm-systems-control-telegram; crew = regime-locus-crew.
WATCHED_SERVICES=ollama,rlm-systems-control-telegram,rlm-master-trader,regime-locus-crew

# ── Offline Ollama advisory (optional — NOT wired to ROEE or Hermes) ────────────
SPOCK_TIMEOUT_SECONDS=90
SPOCK_MIN_CONFIDENCE=0.65
SPOCK_ENABLED=false
SPOCK_OVERRIDE_KEY=

# ── Kirk ────────────────────────────────────────────
DECISION_LOG=${ENTERPRISE_DIR}/data/decisions.jsonl
ENV

chmod 600 "$ENTERPRISE_DIR/config/.env"
log "Config written to $ENTERPRISE_DIR/config/.env"


# ══════════════════════════════════════════════════════════════════════════════
step "8 / 10  Install Enterprise agents"
# ══════════════════════════════════════════════════════════════════════════════

# Copy versioned agents from repo (Hermes crew Scotty/Spock/Kirk are separate — see scripts/run_crew.py).
WATCH_SRC="$REPO_DIR/scripts/rlm_enterprise_watchdog.py"
[[ -f "$WATCH_SRC" ]] || error "Missing $WATCH_SRC — clone Regime-Locus-Matrix at REPO_DIR=$REPO_DIR"
cp "$WATCH_SRC" "$ENTERPRISE_DIR/agents/scotty.py"
cp "$REPO_DIR/scripts/rlm_spock_advisory.py" "$ENTERPRISE_DIR/agents/spock.py"
cp "$REPO_DIR/scripts/rlm_kirk_hook.py" "$ENTERPRISE_DIR/agents/kirk_hook.py"
chmod +x "$ENTERPRISE_DIR/agents/"*.py
log "Enterprise agents installed from repo scripts (watchdog→scotty.py; offline advisory→spock/kirk_hook)"


# ══════════════════════════════════════════════════════════════════════════════
step "9 / 10  Register systemd services"
# ══════════════════════════════════════════════════════════════════════════════

# Scotty service
cat > /etc/systemd/system/scotty.service << SCOSERVICE
[Unit]
Description=RLM host watchdog (Telegram + systemd; Hermes crew is separate)
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
echo "  Hermes AI crew (production Scotty/Spock/Kirk): pip install -e '.[hermes]' && python3 scripts/run_crew.py"
echo "  Host watchdog unit still named scotty.service for compatibility."
echo ""
echo "  Keep TELEGRAM_* secrets only in /opt/enterprise/config/.env — revoke any leaked BotFather tokens."
echo ""
