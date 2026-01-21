#!/bin/bash
# Manual sync script for NostalgiaForInfinityX7 strategy
# Use this script to manually sync the strategy file when needed

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Source and destination paths
SOURCE_FILE="NostalgiaForInfinityX7.py"
DEST_DIR="user_data/strategies/nfi"
DEST_FILE="$DEST_DIR/NostalgiaForInfinityX7.py"

echo -e "${BLUE}[NFI STRATEGY SYNC]${NC} Starting sync process..."

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} Source file not found: $SOURCE_FILE"
    echo "Make sure you're in the repository root directory."
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    echo -e "${YELLOW}[INFO]${NC} Creating destination directory: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

# Show file info
echo -e "${BLUE}[INFO]${NC} Source: $SOURCE_FILE"
echo -e "${BLUE}[INFO]${NC} Destination: $DEST_FILE"

# Check if files are identical
if [ -f "$DEST_FILE" ] && cmp -s "$SOURCE_FILE" "$DEST_FILE"; then
    echo -e "${GREEN}[SUCCESS]${NC} Files are already in sync!"

    # Show version info
    VERSION=$(grep 'return "v' "$DEST_FILE" | head -1 | sed 's/.*return "\(v[^"]*\)".*/\1/' 2>/dev/null)
    if [ -n "$VERSION" ]; then
        echo -e "${GREEN}[INFO]${NC} Current version: $VERSION"
    fi

    exit 0
fi

# Show differences if destination exists
if [ -f "$DEST_FILE" ]; then
    echo -e "${YELLOW}[WARNING]${NC} Files differ. Showing version comparison:"

    SRC_VERSION=$(grep 'return "v' "$SOURCE_FILE" | head -1 | sed 's/.*return "\(v[^"]*\)".*/\1/' 2>/dev/null)
    DEST_VERSION=$(grep 'return "v' "$DEST_FILE" | head -1 | sed 's/.*return "\(v[^"]*\)".*/\1/' 2>/dev/null)

    echo -e "  Source version: ${GREEN}$SRC_VERSION${NC}"
    echo -e "  Destination version: ${YELLOW}$DEST_VERSION${NC}"
fi

# Perform the copy
echo -e "${BLUE}[SYNCING]${NC} Copying $SOURCE_FILE to $DEST_FILE..."
if cp "$SOURCE_FILE" "$DEST_FILE"; then
    echo -e "${GREEN}[SUCCESS]${NC} Successfully synced strategy file!"

    # Show new version info
    VERSION=$(grep 'return "v' "$DEST_FILE" | head -1 | sed 's/.*return "\(v[^"]*\)".*/\1/' 2>/dev/null)
    if [ -n "$VERSION" ]; then
        echo -e "${GREEN}[INFO]${NC} Updated to version: $VERSION"
    fi

    # Check file size
    SIZE=$(ls -lh "$DEST_FILE" | awk '{print $5}')
    echo -e "${GREEN}[INFO]${NC} File size: $SIZE"

    exit 0
else
    echo -e "${RED}[ERROR]${NC} Failed to sync strategy file!"
    exit 1
fi
