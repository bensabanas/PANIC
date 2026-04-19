#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# PANIC Uninstaller
# Removes the PANIC plugin from OpenClaw.
#
# Usage:
#   cd /path/to/panic && bash uninstall.sh
#   bash uninstall.sh --keep-profiles   # keep memory data
#   bash uninstall.sh --purge           # remove everything including venv
# ─────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

PANIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEEP_PROFILES=false
PURGE=false

for arg in "$@"; do
  case "$arg" in
    --keep-profiles) KEEP_PROFILES=true ;;
    --purge) PURGE=true ;;
  esac
done

info()  { echo -e "${BLUE}▸${RESET} $*"; }
ok()    { echo -e "${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${YELLOW}⚠${RESET} $*"; }

echo -e "\n${BOLD}PANIC Uninstaller${RESET}\n"

# Remove from OpenClaw config
OPENCLAW_JSON="${OPENCLAW_CONFIG:-$HOME/.openclaw/openclaw.json}"
if [[ -f "$OPENCLAW_JSON" ]]; then
  python3 - "$OPENCLAW_JSON" "$PANIC_ROOT" << 'PYEOF'
import sys, json

config_path = sys.argv[1]
panic_root = sys.argv[2]

with open(config_path) as f:
    config = json.load(f)

plugins = config.get("plugins", {})

# Remove entry
plugins.get("entries", {}).pop("panic", None)

# Remove install record
plugins.get("installs", {}).pop("panic", None)

# Clear context engine slot if it's panic
if plugins.get("slots", {}).get("contextEngine") == "panic":
    plugins["slots"].pop("contextEngine")

# Remove load path
plugin_path = panic_root + "/plugin"
paths = plugins.get("load", {}).get("paths", [])
plugins.setdefault("load", {})["paths"] = [p for p in paths if p != plugin_path]

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("Config cleaned")
PYEOF
  ok "Removed PANIC from OpenClaw config"
else
  warn "OpenClaw config not found"
fi

# Remove profiles
if [[ "$KEEP_PROFILES" == false ]]; then
  PROFILE_DIR="$HOME/.openclaw/panic"
  if [[ -d "$PROFILE_DIR" ]]; then
    echo -e ""
    echo -e "${YELLOW}This will delete all PANIC memory data:${RESET}"
    echo -e "  $PROFILE_DIR"
    echo -e ""
    read -p "Delete memory profiles? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
      rm -rf "$PROFILE_DIR"
      ok "Profiles deleted"
    else
      info "Profiles kept at $PROFILE_DIR"
    fi
  fi
else
  info "Profiles preserved (--keep-profiles)"
fi

# Purge venv and build artifacts
if [[ "$PURGE" == true ]]; then
  [[ -d "$PANIC_ROOT/.venv" ]] && rm -rf "$PANIC_ROOT/.venv" && ok "Virtual environment removed"
  [[ -d "$PANIC_ROOT/plugin/dist" ]] && rm -rf "$PANIC_ROOT/plugin/dist" && ok "Plugin build artifacts removed"
  [[ -d "$PANIC_ROOT/plugin/node_modules" ]] && rm -rf "$PANIC_ROOT/plugin/node_modules" && ok "Node modules removed"
fi

# Restart gateway
info "Restarting OpenClaw gateway..."
openclaw gateway restart 2>/dev/null && ok "Gateway restarted" || warn "Restart manually: openclaw gateway restart"

echo -e "\n${GREEN}${BOLD}PANIC uninstalled.${RESET}\n"
if [[ "$KEEP_PROFILES" == true ]] || [[ -d "$HOME/.openclaw/panic" ]]; then
  echo -e "  ${DIM}Memory data preserved at:${RESET} ~/.openclaw/panic"
  echo -e "  ${DIM}Reinstall with:${RESET} bash install.sh"
fi
echo ""
