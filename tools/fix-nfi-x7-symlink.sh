#!/bin/bash

###############################################################################
# Fix NostalgiaForInfinityX7 Symlink
###############################################################################
# This script fixes the symlink for NostalgiaForInfinityX7 strategy
# Usage: ./fix-nfi-x7-symlink.sh
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO_PATH="/opt/trading/nfi-custom-strategies"
STRATEGY_NAME="NostalgiaForInfinityX7"

echo -e "${BLUE}=========================================================================="
echo -e "Fixing ${STRATEGY_NAME} Symlink"
echo -e "==========================================================================${NC}"
echo ""

# Change to repo directory
if [ ! -d "$REPO_PATH" ]; then
    echo -e "${RED}[ERROR]${NC} Repository path does not exist: $REPO_PATH"
    exit 1
fi

cd "$REPO_PATH" || exit 1

# Check if main file exists
MAIN_FILE="${REPO_PATH}/${STRATEGY_NAME}.py"
if [ ! -f "$MAIN_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} Main strategy file not found: $MAIN_FILE"
    echo ""
    echo "Available .py files in repo root:"
    ls -1 *.py 2>/dev/null || echo "  No .py files found"
    exit 1
fi

echo -e "${GREEN}✓${NC} Main file exists: $MAIN_FILE"
echo ""

# Create directory if it doesn't exist
SYMLINK_DIR="${REPO_PATH}/user_data/strategies/nfi"
if [ ! -d "$SYMLINK_DIR" ]; then
    echo -e "${YELLOW}[INFO]${NC} Creating directory: $SYMLINK_DIR"
    mkdir -p "$SYMLINK_DIR"
fi

# Remove existing symlink or file if it exists
SYMLINK_PATH="${SYMLINK_DIR}/${STRATEGY_NAME}.py"
if [ -e "$SYMLINK_PATH" ] || [ -L "$SYMLINK_PATH" ]; then
    echo -e "${YELLOW}[INFO]${NC} Removing existing symlink/file: $SYMLINK_PATH"
    rm -f "$SYMLINK_PATH"
fi

# Create new symlink
echo -e "${BLUE}[INFO]${NC} Creating symlink..."
ln -s "../../../${STRATEGY_NAME}.py" "$SYMLINK_PATH"

# Verify symlink
if [ -L "$SYMLINK_PATH" ] && [ -e "$SYMLINK_PATH" ]; then
    echo -e "${GREEN}✓${NC} Symlink created successfully!"
    echo ""
    echo "Details:"
    ls -lh "$SYMLINK_PATH"
    echo ""
    echo "Target file size:"
    ls -lh "$MAIN_FILE" | awk '{print $5, $9}'
else
    echo -e "${RED}[ERROR]${NC} Failed to create valid symlink"
    exit 1
fi

echo ""
echo -e "${BLUE}[INFO]${NC} Testing Python syntax..."
python3 -m py_compile "$MAIN_FILE" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} No Python syntax errors found"
else
    echo -e "${RED}✗${NC} Python syntax errors detected!"
    echo ""
    echo "Please check the file for errors before restarting the container."
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================================================="
echo -e "Symlink fixed successfully!"
echo -e "==========================================================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Restart the container:"
echo "   cd $REPO_PATH"
echo "   ./deploy-multi-strategies.sh restart nfi-x7"
echo ""
echo "2. Check the logs:"
echo "   ./deploy-multi-strategies.sh logs nfi-x7"
echo ""
