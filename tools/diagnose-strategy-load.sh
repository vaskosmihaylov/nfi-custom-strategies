#!/bin/bash

###############################################################################
# Strategy Loading Diagnostic Script
###############################################################################
# This script helps diagnose why a strategy fails to load in Freqtrade
# Usage: ./diagnose-strategy-load.sh [strategy-name]
# Example: ./diagnose-strategy-load.sh NostalgiaForInfinityX7
###############################################################################

set +e  # Don't exit on error - we want to see all diagnostics

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
STRATEGY_NAME="${1:-NostalgiaForInfinityX7}"
REPO_PATH="${REPO_PATH:-/opt/trading/nfi-custom-strategies}"

echo -e "${BLUE}=========================================================================="
echo -e "Strategy Loading Diagnostic Tool"
echo -e "==========================================================================${NC}"
echo ""
echo -e "${CYAN}Strategy:${NC} $STRATEGY_NAME"
echo -e "${CYAN}Repo Path:${NC} $REPO_PATH"
echo ""

# Change to repo directory
if [ ! -d "$REPO_PATH" ]; then
    echo -e "${RED}[ERROR]${NC} Repository path does not exist: $REPO_PATH"
    exit 1
fi

cd "$REPO_PATH" || exit 1

echo -e "${BLUE}=== 1. Checking File Locations ===${NC}"
echo ""

# Check main strategy file
MAIN_FILE="$REPO_PATH/${STRATEGY_NAME}.py"
if [ -f "$MAIN_FILE" ]; then
    echo -e "${GREEN}✓${NC} Main strategy file exists: $MAIN_FILE"
    ls -lh "$MAIN_FILE"
else
    echo -e "${RED}✗${NC} Main strategy file NOT found: $MAIN_FILE"
fi
echo ""

# Check symlink location
SYMLINK_PATH="$REPO_PATH/user_data/strategies/nfi/${STRATEGY_NAME}.py"
if [ -L "$SYMLINK_PATH" ]; then
    echo -e "${GREEN}✓${NC} Symlink exists: $SYMLINK_PATH"
    ls -lh "$SYMLINK_PATH"

    # Check if symlink is valid
    if [ -e "$SYMLINK_PATH" ]; then
        echo -e "${GREEN}✓${NC} Symlink is valid (points to existing file)"
    else
        echo -e "${RED}✗${NC} Symlink is BROKEN (points to non-existent file)"
        echo -e "  Target: $(readlink "$SYMLINK_PATH")"
    fi
else
    echo -e "${RED}✗${NC} Symlink NOT found: $SYMLINK_PATH"
    if [ -f "$SYMLINK_PATH" ]; then
        echo -e "${YELLOW}!${NC} Regular file exists at symlink location (not a symlink)"
        ls -lh "$SYMLINK_PATH"
    fi
fi
echo ""

# Check if file is accessible inside docker container
echo -e "${BLUE}=== 2. Checking File in Docker Container ===${NC}"
echo ""
CONTAINER_NAME="freqtrade-nfi-x7"
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}✓${NC} Container $CONTAINER_NAME is running"

    # Check if file exists in container
    echo ""
    echo "Checking file in container at: /freqtrade/user_data/strategies/nfi/${STRATEGY_NAME}.py"
    docker exec "$CONTAINER_NAME" ls -lh "/freqtrade/user_data/strategies/nfi/${STRATEGY_NAME}.py" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} File accessible in container"
    else
        echo -e "${RED}✗${NC} File NOT accessible in container"
    fi
else
    echo -e "${YELLOW}!${NC} Container $CONTAINER_NAME is not running"
fi
echo ""

echo -e "${BLUE}=== 3. Checking Python Syntax ===${NC}"
echo ""

# Check for syntax errors using Python
if [ -f "$MAIN_FILE" ]; then
    echo "Running Python syntax check..."
    python3 -m py_compile "$MAIN_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} No Python syntax errors found"
    else
        echo -e "${RED}✗${NC} Python syntax errors detected"
    fi
else
    echo -e "${YELLOW}!${NC} Cannot check syntax - file not found"
fi
echo ""

echo -e "${BLUE}=== 4. Checking Class Definition ===${NC}"
echo ""

# Check if class exists with correct name
if [ -f "$MAIN_FILE" ]; then
    echo "Searching for class definition..."
    CLASS_LINE=$(grep -n "^class ${STRATEGY_NAME}" "$MAIN_FILE" | head -1)
    if [ -n "$CLASS_LINE" ]; then
        echo -e "${GREEN}✓${NC} Class '$STRATEGY_NAME' found:"
        echo "  $CLASS_LINE"
    else
        echo -e "${RED}✗${NC} Class '$STRATEGY_NAME' NOT found"
        echo ""
        echo "Available classes in file:"
        grep -n "^class " "$MAIN_FILE" || echo "  No classes found"
    fi
else
    echo -e "${YELLOW}!${NC} Cannot check class - file not found"
fi
echo ""

echo -e "${BLUE}=== 5. Checking Import Errors ===${NC}"
echo ""

# Try to import the strategy in Docker container
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Testing import in container..."
    docker exec "$CONTAINER_NAME" python3 -c "
import sys
sys.path.insert(0, '/freqtrade/user_data/strategies/nfi')
try:
    from ${STRATEGY_NAME} import ${STRATEGY_NAME}
    print('✓ Successfully imported ${STRATEGY_NAME}')
except ImportError as e:
    print('✗ Import Error:', str(e))
except Exception as e:
    print('✗ Error:', str(e))
" 2>&1
else
    echo -e "${YELLOW}!${NC} Cannot test import - container not running"
fi
echo ""

echo -e "${BLUE}=== 6. Checking Recent File Changes ===${NC}"
echo ""

if [ -f "$MAIN_FILE" ]; then
    echo "Last modified:"
    stat -c '%y %n' "$MAIN_FILE" 2>/dev/null || stat -f '%Sm %N' "$MAIN_FILE" 2>/dev/null
    echo ""

    echo "Recent Git changes:"
    git log --oneline --follow -5 "${STRATEGY_NAME}.py" 2>/dev/null || echo "No git history available"
else
    echo -e "${YELLOW}!${NC} Cannot check changes - file not found"
fi
echo ""

echo -e "${BLUE}=== 7. Checking Container Logs ===${NC}"
echo ""

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Recent container logs (last 30 lines):"
    echo "----------------------------------------------------------------------"
    docker logs --tail 30 "$CONTAINER_NAME" 2>&1 | grep -A 5 -B 5 -i "error\|exception\|impossible to load"
    echo "----------------------------------------------------------------------"
else
    echo -e "${YELLOW}!${NC} Container not running - cannot check logs"
fi
echo ""

echo -e "${BLUE}=== Diagnostic Summary ===${NC}"
echo ""

# Summary checks
ISSUES_FOUND=0

[ ! -f "$MAIN_FILE" ] && echo -e "${RED}✗${NC} Main file missing" && ((ISSUES_FOUND++))
[ ! -L "$SYMLINK_PATH" ] && echo -e "${RED}✗${NC} Symlink missing" && ((ISSUES_FOUND++))
[ -L "$SYMLINK_PATH" ] && [ ! -e "$SYMLINK_PATH" ] && echo -e "${RED}✗${NC} Symlink broken" && ((ISSUES_FOUND++))

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}No obvious issues found with file structure${NC}"
    echo ""
    echo -e "${YELLOW}Recommendations:${NC}"
    echo "1. Check container logs above for detailed error messages"
    echo "2. Verify all Python dependencies are installed in container"
    echo "3. Try restarting the container: docker restart $CONTAINER_NAME"
    echo "4. Check if there are any import conflicts with other strategies"
else
    echo -e "${RED}Found $ISSUES_FOUND issue(s) with file structure${NC}"
    echo ""
    echo -e "${YELLOW}Recommended fixes:${NC}"

    if [ ! -f "$MAIN_FILE" ]; then
        echo "1. Copy strategy file to: $MAIN_FILE"
    fi

    if [ ! -L "$SYMLINK_PATH" ] || [ ! -e "$SYMLINK_PATH" ]; then
        echo "2. Create/fix symlink:"
        echo "   mkdir -p $(dirname "$SYMLINK_PATH")"
        echo "   ln -sf ../../../${STRATEGY_NAME}.py $SYMLINK_PATH"
    fi

    echo "3. Restart container: docker restart $CONTAINER_NAME"
fi

echo ""
echo -e "${BLUE}=========================================================================="
echo -e "Diagnostic Complete"
echo -e "==========================================================================${NC}"
