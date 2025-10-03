#!/bin/bash
# FVG Strategy Fix - Deployment Script
# This script safely deploys the fixed FVG strategy with automatic backup

set -e  # Exit on any error

echo "=========================================="
echo "FVG Strategy V2 - DEPLOYMENT SCRIPT"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

STRATEGY_DIR="user_data/strategies/FVGAdvancedStrategy_V2"
CURRENT_STRATEGY="${STRATEGY_DIR}/FVGAdvancedStrategy_V2.py"
FIXED_STRATEGY="${STRATEGY_DIR}/FVGAdvancedStrategy_V2_FIXED.py"
BACKUP_DIR="${STRATEGY_DIR}/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Step 1: Verify we're in the right directory
echo "Step 1: Verifying project directory..."
if [ ! -d "user_data/strategies" ]; then
    echo -e "${RED}ERROR: Not in NostalgiaForInfinity project root directory!${NC}"
    echo "Please run this script from: /Users/vasko.mihaylov/AI_Projects/NostalgiaForInfinity"
    exit 1
fi
echo -e "${GREEN}✓ Project directory verified${NC}"
echo ""

# Step 2: Check if fixed strategy exists
echo "Step 2: Checking for fixed strategy file..."
if [ ! -f "${FIXED_STRATEGY}" ]; then
    echo -e "${RED}ERROR: Fixed strategy file not found!${NC}"
    echo "Expected location: ${FIXED_STRATEGY}"
    exit 1
fi
echo -e "${GREEN}✓ Fixed strategy file found${NC}"
echo ""

# Step 3: Create backup directory
echo "Step 3: Creating backup directory..."
mkdir -p "${BACKUP_DIR}"
echo -e "${GREEN}✓ Backup directory ready${NC}"
echo ""

# Step 4: Backup current strategy
echo "Step 4: Backing up current strategy..."
if [ -f "${CURRENT_STRATEGY}" ]; then
    BACKUP_FILE="${BACKUP_DIR}/FVGAdvancedStrategy_V2_BACKUP_${TIMESTAMP}.py"
    cp "${CURRENT_STRATEGY}" "${BACKUP_FILE}"
    echo -e "${GREEN}✓ Current strategy backed up to: ${BACKUP_FILE}${NC}"
else
    echo -e "${YELLOW}⚠ No current strategy file found (first deployment?)${NC}"
fi
echo ""

# Step 5: Deploy fixed strategy
echo "Step 5: Deploying fixed strategy..."
cp "${FIXED_STRATEGY}" "${CURRENT_STRATEGY}"
echo -e "${GREEN}✓ Fixed strategy deployed successfully${NC}"
echo ""

# Step 6: Verify deployment
echo "Step 6: Verifying deployment..."
if grep -q "FVG Advanced Strategy V2 - FIXED VERSION" "${CURRENT_STRATEGY}"; then
    echo -e "${GREEN}✓ Fixed version confirmed in deployed file${NC}"
else
    echo -e "${RED}ERROR: Deployed file doesn't contain expected fixed version marker!${NC}"
    echo "Rolling back..."
    if [ -f "${BACKUP_FILE}" ]; then
        cp "${BACKUP_FILE}" "${CURRENT_STRATEGY}"
        echo -e "${YELLOW}Rolled back to previous version${NC}"
    fi
    exit 1
fi
echo ""

# Step 7: Stop running bot
echo "Step 7: Stopping FVG bot container..."
echo -e "${YELLOW}Running: ./deploy-multi-strategies.sh stop fvg${NC}"
if ./deploy-multi-strategies.sh stop fvg 2>/dev/null; then
    echo -e "${GREEN}✓ Bot stopped${NC}"
else
    echo -e "${YELLOW}⚠ Bot might not be running (this is OK for first deployment)${NC}"
fi
echo ""

# Step 8: Remove old container
echo "Step 8: Removing old container..."
if docker ps -a | grep -q freqtrade-fvg; then
    docker rm freqtrade-fvg 2>/dev/null || true
    echo -e "${GREEN}✓ Old container removed${NC}"
else
    echo -e "${YELLOW}⚠ No existing container found (this is OK for first deployment)${NC}"
fi
echo ""

# Step 9: Start bot with new strategy
echo "Step 9: Starting FVG bot with fixed strategy..."
echo -e "${YELLOW}Running: ./deploy-multi-strategies.sh start fvg${NC}"
if ./deploy-multi-strategies.sh start fvg; then
    echo -e "${GREEN}✓ Bot started successfully${NC}"
else
    echo -e "${RED}ERROR: Failed to start bot!${NC}"
    echo "Check the logs with: docker logs freqtrade-fvg"
    exit 1
fi
echo ""

# Step 10: Display monitoring instructions
echo "=========================================="
echo "DEPLOYMENT SUCCESSFUL! ✓"
echo "=========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Monitor the logs:"
echo "   docker logs -f freqtrade-fvg"
echo ""
echo "2. Check for trades in the first 6 hours:"
echo "   - Expected: 1-3 trades should appear"
echo "   - Look for 'FVG Advanced Strategy V2 - FIXED VERSION initialized' in logs"
echo ""
echo "3. Verify improvements:"
echo "   - Trades should start opening (vs 0 trades before)"
echo "   - Stop losses should be tighter (-9% vs -30%)"
echo "   - Partial exits should execute at 5%, 8%, 12% profit"
echo ""
echo "4. Monitor key metrics over 24 hours:"
echo "   - Trade frequency: Target 5-15 trades/day"
echo "   - Win rate: Target >50%"
echo "   - Max loss per trade: Should be <9%"
echo ""
echo "Documentation:"
echo "   - Full analysis: ${STRATEGY_DIR}/FVG_COMPLETE_FIX_ANALYSIS.md"
echo "   - Backup location: ${BACKUP_FILE}"
echo ""
echo "Troubleshooting:"
echo "   If no trades in 6 hours, see troubleshooting section in analysis doc"
echo ""
echo "=========================================="
echo "Happy Trading! 🚀"
echo "=========================================="
