#!/bin/bash

# --- Configuration (all values come from environment variables set in docker-compose.yml) ---

STRATEGY="${STRATEGY:-NostalgiaForInfinityX7}"
EXCHANGE="${EXCHANGE:-binance}"

# Project root is mounted at /data inside this container
BASE_DIR="/data"
STRATEGY_FILE="$BASE_DIR/${STRATEGY}.py"
CONFIGS_DIR="$BASE_DIR/configs"

# Pairlist filename — override via NFI_PAIRLIST_FILE if your exchange uses a different name
PAIRLIST_FILE="${NFI_PAIRLIST_FILE:-pairlist-volume-${EXCHANGE}-usdt.json}"

TEMP_DIR="/tmp/nfi_update"
REPO_URL="https://raw.githubusercontent.com/iterativv/NostalgiaForInfinity/main"

mkdir -p "$TEMP_DIR" "$CONFIGS_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "------------------------------------------------"
echo "Date: $(date)"
echo "Checking for NFI updates..."

CHANGES_DETECTED=false

# --- Downloads a file from GitHub and replaces the local copy if it changed ---
check_and_update() {
    local remote_suffix=$1
    local local_path=$2
    local filename
    filename=$(basename "$local_path")
    local remote_url="$REPO_URL/$remote_suffix"
    local tmp_file="$TEMP_DIR/$filename"

    echo -n "Checking $filename ... "

    if ! wget -q -O "$tmp_file" "$remote_url"; then
        echo -e "${RED}[ERROR] Download failed for $remote_url${NC}"
        return
    fi

    if [ -f "$local_path" ]; then
        LOCAL_HASH=$(md5sum "$local_path" | awk '{print $1}')
        REMOTE_HASH=$(md5sum "$tmp_file"   | awk '{print $1}')

        if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
            echo -e "${GREEN}[UPDATED]${NC}"
            # cp instead of mv — avoids cross-filesystem issues with Docker volumes on Windows
            cp "$tmp_file" "$local_path"
            CHANGES_DETECTED=true
        else
            echo -e "${YELLOW}[OK] Up to date${NC}"
        fi
    else
        echo -e "${GREEN}[NEW FILE]${NC}"
        cp "$tmp_file" "$local_path"
        CHANGES_DETECTED=true
    fi

    rm -f "$tmp_file"
}

# --- Check all three files ---
check_and_update "${STRATEGY}.py"                      "$STRATEGY_FILE"
check_and_update "configs/blacklist-${EXCHANGE}.json"  "$CONFIGS_DIR/blacklist-${EXCHANGE}.json"
check_and_update "configs/${PAIRLIST_FILE}"            "$CONFIGS_DIR/${PAIRLIST_FILE}"

echo "------------------------------------------------"

if [ "$CHANGES_DETECTED" = true ]; then
    echo -e "${GREEN}Updates applied! Restarting freqtrade...${NC}"
    rm -rf "$TEMP_DIR"

    # Restart via Docker Compose using the mounted socket.
    # COMPOSE_PROJECT_NAME must match the name Docker Compose assigned to your stack
    # (defaults to the lowercase folder name — check with: docker compose ls)
    COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-nostalgiaforinfinity}" \
        docker compose -f /data/docker-compose.yml restart freqtrade

    echo "Bot restarted successfully."
else
    echo -e "${GREEN}No updates found. System is up to date.${NC}"
    rm -rf "$TEMP_DIR"
fi
