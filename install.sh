#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# PANIC Installer
# Installs the PANIC memory engine as an OpenClaw plugin.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/bensabanas/panic/main/install.sh | bash
#   — or —
#   cd /path/to/panic && bash install.sh
#
# What it does:
#   1. Checks prerequisites (Python 3.11+, Node.js, OpenClaw)
#   2. Creates Python virtual environment & installs dependencies
#   3. Downloads the sentence-transformers encoder model
#   4. Builds the TypeScript plugin
#   5. Installs plugin into OpenClaw (linked)
#   6. Configures OpenClaw (plugin entry + context engine slot)
#   7. Creates the default profile
#   8. Restarts the gateway
# ─────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

PANIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_PORT="${PANIC_PORT:-7420}"
PROFILE="${PANIC_PROFILE:-default}"

info()  { echo -e "${BLUE}▸${RESET} $*"; }
ok()    { echo -e "${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${YELLOW}⚠${RESET} $*"; }
fail()  { echo -e "${RED}✗${RESET} $*"; exit 1; }
step()  { echo -e "\n${BOLD}[$1/$TOTAL_STEPS] $2${RESET}"; }

TOTAL_STEPS=8

echo -e "\n${BOLD}${RED}P${YELLOW}A${GREEN}N${BLUE}I${RED}C${RESET} ${DIM}— Persistent Memory Engine Installer${RESET}\n"

# ── Step 1: Prerequisites ──
step 1 "Checking prerequisites"

# Python
if command -v python3 &>/dev/null; then
  PY="python3"
elif command -v python &>/dev/null; then
  PY="python"
else
  fail "Python 3.11+ is required but not found. Install it first."
fi

PY_VER=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PY -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PY -c "import sys; print(sys.version_info.minor)")
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11 ]]; then
  fail "Python $PY_VER found but 3.11+ is required."
fi
ok "Python $PY_VER"

# Node.js
if ! command -v node &>/dev/null; then
  fail "Node.js is required but not found. Install it first."
fi
NODE_VER=$(node -v)
ok "Node.js $NODE_VER"

# npm
if ! command -v npm &>/dev/null; then
  fail "npm is required but not found."
fi
ok "npm $(npm -v 2>/dev/null)"

# OpenClaw
if ! command -v openclaw &>/dev/null; then
  fail "OpenClaw is required but not found. Install it: npm install -g openclaw"
fi
OC_VER=$(openclaw --version 2>/dev/null | head -1 || echo "unknown")
ok "OpenClaw $OC_VER"

# ── Step 2: Python virtual environment ──
step 2 "Setting up Python virtual environment"

VENV_DIR="$PANIC_ROOT/.venv"
if [[ -d "$VENV_DIR" ]]; then
  info "Virtual environment already exists at $VENV_DIR"
else
  $PY -m venv "$VENV_DIR"
  ok "Created virtual environment"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Upgrade pip
"$VENV_PIP" install --quiet --upgrade pip

# ── Step 3: Install Python dependencies ──
step 3 "Installing Python dependencies"

# Runtime requirements only
"$VENV_PIP" install --quiet \
  "numpy>=1.24.0" \
  "scipy>=1.11.0" \
  "torch>=2.0.0" \
  "sentence-transformers>=2.2.0" \
  "spacy>=3.6.0" \
  "tiktoken>=0.5.0" \
  "litellm>=1.0.0" \
  "fastapi>=0.104.0" \
  "uvicorn>=0.24.0" \
  "pydantic>=2.0.0" \
  "httpx>=0.25.0" \
  "python-multipart>=0.0.6"

ok "Python dependencies installed"

# ── Step 4: Download encoder model ──
step 4 "Downloading sentence-transformers encoder model"

"$VENV_PY" -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print('Model loaded:', m.get_sentence_embedding_dimension(), 'dimensions')
" 2>/dev/null && ok "Encoder model ready (all-MiniLM-L6-v2, 384d)" || warn "Model download may have failed — will retry on first use"

# ── Step 5: Build TypeScript plugin ──
step 5 "Building TypeScript plugin"

cd "$PANIC_ROOT/plugin"
if [[ ! -d "node_modules" ]]; then
  npm install --silent 2>/dev/null
fi
npm run build --silent 2>/dev/null
ok "Plugin built"
cd "$PANIC_ROOT"

# ── Step 6: Install plugin into OpenClaw ──
step 6 "Installing plugin into OpenClaw"

# Use --link so the plugin runs from source (no copy)
# --force in case it's already installed
openclaw plugins install -l "$PANIC_ROOT/plugin" --force 2>/dev/null && \
  ok "Plugin installed (linked)" || \
  warn "Plugin install command failed — may already be configured"

# ── Step 7: Configure OpenClaw ──
step 7 "Configuring OpenClaw"

# Use openclaw's config CLI if available, otherwise patch JSON directly
OPENCLAW_JSON="${OPENCLAW_CONFIG:-$HOME/.openclaw/openclaw.json}"

if [[ ! -f "$OPENCLAW_JSON" ]]; then
  fail "OpenClaw config not found at $OPENCLAW_JSON"
fi

# Patch the config using Python for reliability
"$PY" - "$OPENCLAW_JSON" "$PANIC_ROOT" "$VENV_PY" "$SIDECAR_PORT" "$PROFILE" << 'PYEOF'
import sys, json, os

config_path = sys.argv[1]
panic_root = sys.argv[2]
python_path = sys.argv[3]
port = int(sys.argv[4])
profile = sys.argv[5]

with open(config_path) as f:
    config = json.load(f)

plugins = config.setdefault("plugins", {})
entries = plugins.setdefault("entries", {})
installs = plugins.setdefault("installs", {})
slots = plugins.setdefault("slots", {})
load = plugins.setdefault("load", {})
paths = load.setdefault("paths", [])

# Plugin entry with sidecar config
entries["panic"] = {
    "enabled": True,
    "config": {
        "profile": profile,
        "sidecarPort": port,
        "pythonPath": python_path,
        "panicRoot": panic_root,
    }
}

# Context engine slot
slots["contextEngine"] = "panic"

# Load path
plugin_path = os.path.join(panic_root, "plugin")
if plugin_path not in paths:
    paths.append(plugin_path)

# Install record
installs["panic"] = {
    "source": "path",
    "sourcePath": plugin_path,
    "installPath": plugin_path,
    "version": "0.1.0",
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("Config updated")
PYEOF

ok "OpenClaw configured (context engine: panic, port: $SIDECAR_PORT)"

# ── Step 8: Create default profile & restart ──
step 8 "Finalizing"

# Create profile directory
PROFILE_DIR="$HOME/.openclaw/panic/profiles/$PROFILE"
mkdir -p "$PROFILE_DIR"/{episodes,semantic,procedural,graphs}

# Initialize memory files if they don't exist
for f in semantic/entities.md semantic/facts.md semantic/preferences.md; do
  target="$PROFILE_DIR/$f"
  if [[ ! -f "$target" ]]; then
    basename="${f##*/}"
    title="${basename%.md}"
    title="$(echo "$title" | sed 's/./\U&/')"  # capitalize first letter
    echo "# $title" > "$target"
    echo "" >> "$target"
  fi
done

for f in procedural/workflows.md procedural/failures.md; do
  target="$PROFILE_DIR/$f"
  if [[ ! -f "$target" ]]; then
    basename="${f##*/}"
    title="${basename%.md}"
    title="$(echo "$title" | sed 's/./\U&/')"
    echo "# $title" > "$target"
    echo "" >> "$target"
  fi
done

ok "Profile '$PROFILE' initialized"

# Restart gateway
info "Restarting OpenClaw gateway..."
openclaw gateway restart 2>/dev/null && ok "Gateway restarted" || warn "Gateway restart failed — restart manually with: openclaw gateway restart"

# ── Done ──
echo -e "\n${GREEN}${BOLD}Installation complete!${RESET}\n"
echo -e "  ${DIM}Dashboard:${RESET}  http://127.0.0.1:$SIDECAR_PORT"
echo -e "  ${DIM}Profile:${RESET}    $PROFILE_DIR"
echo -e "  ${DIM}Plugin:${RESET}     $PANIC_ROOT/plugin"
echo -e "  ${DIM}Sidecar:${RESET}    port $SIDECAR_PORT"
echo -e ""
echo -e "  ${DIM}Verify:${RESET}     curl http://127.0.0.1:$SIDECAR_PORT/api/status"
echo -e "  ${DIM}Logs:${RESET}       openclaw gateway logs | grep panic"
echo -e ""
echo -e "  ${YELLOW}Tip:${RESET} Open the dashboard to import existing memory files"
echo -e "       or use the Memory Editor to add knowledge manually."
echo -e ""
