#!/bin/bash
# Production Trading Data Reset Script
# Safely wipes all trading data while preserving configurations
#
# IMPORTANT: This will delete:
# - All trade history (SQLite databases)
# - All logs
# - All open positions (closes them first)
#
# This will KEEP:
# - Strategy files
# - Configuration files
# - Environment files
# - Docker setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Check if running in production directory
if [ ! -f "docker-compose-multi-strategies.yml" ]; then
    print_error "docker-compose-multi-strategies.yml not found!"
    print_error "Please run this script from: /opt/trading/nfi-custom-strategies"
    exit 1
fi

print_header "🧹 Production Trading Data Reset"

print_warning "⚠️  WARNING: This will delete ALL trading data!"
print_warning "This includes:"
echo "  - All trade history (SQLite databases)"
echo "  - All logs"
echo "  - Any open positions will be closed"
echo ""
print_status "This will KEEP:"
echo "  - Strategy files"
echo "  - Configuration files (env-files/)"
echo "  - Docker setup"
echo ""

# Confirm action
read -p "Are you sure you want to continue? Type 'YES' to confirm: " -r
echo
if [[ ! $REPLY == "YES" ]]; then
    print_error "Reset cancelled"
    exit 1
fi

# Create backup directory with timestamp
BACKUP_DIR="backups/reset-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

print_header "Step 1: Closing All Open Positions"

print_status "Getting list of running containers..."
CONTAINERS=$(docker compose -f docker-compose-multi-strategies.yml ps --services --filter "status=running" 2>/dev/null || true)

if [ -n "$CONTAINERS" ]; then
    print_status "Found running containers, attempting to close positions..."

    # Try to force-exit all open positions via API
    for container in $CONTAINERS; do
        PORT=$(docker compose -f docker-compose-multi-strategies.yml port $container 8080 2>/dev/null | cut -d: -f2 || echo "")
        if [ -n "$PORT" ]; then
            print_status "Closing positions for $container (port $PORT)..."

            # Try to force-exit all trades
            curl -s -X POST "http://127.0.0.1:$PORT/api/v1/forceexit/all" \
                -H "Content-Type: application/json" \
                -d '{}' > /dev/null 2>&1 || print_warning "Could not close positions for $container"

            sleep 2
        fi
    done

    print_success "Position closing attempted (check manually if needed)"
else
    print_status "No running containers found"
fi

print_header "Step 2: Stopping All Containers"

print_status "Stopping all FreqTrade containers..."
if docker compose -f docker-compose-multi-strategies.yml down; then
    print_success "All containers stopped"
else
    print_error "Failed to stop containers"
    exit 1
fi

print_header "Step 3: Backing Up Current Data"

print_status "Backing up to: $BACKUP_DIR"

# Backup SQLite databases
if ls user_data/*.sqlite 2>/dev/null; then
    mkdir -p "$BACKUP_DIR/databases"
    cp user_data/*.sqlite "$BACKUP_DIR/databases/" 2>/dev/null || true
    print_success "Backed up $(ls user_data/*.sqlite 2>/dev/null | wc -l) database files"
fi

# Backup logs
if [ -d "user_data/logs" ]; then
    mkdir -p "$BACKUP_DIR/logs"
    cp -r user_data/logs/* "$BACKUP_DIR/logs/" 2>/dev/null || true
    print_success "Backed up logs"
fi

# Create manifest
cat > "$BACKUP_DIR/MANIFEST.txt" << EOF
Trading Data Backup
Created: $(date)
Hostname: $(hostname)

This backup was created before resetting all trading data.

Contents:
- databases/ : All SQLite trade databases
- logs/      : All trading logs

To restore (NOT RECOMMENDED):
1. Stop containers: docker compose -f docker-compose-multi-strategies.yml down
2. Restore databases: cp $BACKUP_DIR/databases/*.sqlite user_data/
3. Restart: docker compose -f docker-compose-multi-strategies.yml up -d
EOF

print_success "Backup manifest created: $BACKUP_DIR/MANIFEST.txt"

print_header "Step 4: Deleting Trading Data"

# Delete SQLite databases
print_status "Deleting trade databases..."
DELETED_DBS=0
for db in user_data/*.sqlite; do
    if [ -f "$db" ]; then
        rm -f "$db"
        ((DELETED_DBS++))
        print_status "Deleted: $(basename $db)"
    fi
done
print_success "Deleted $DELETED_DBS database files"

# Delete logs
print_status "Deleting logs..."
if [ -d "user_data/logs" ]; then
    rm -rf user_data/logs/*
    print_success "Deleted all logs"
fi

# Delete any .lock files
print_status "Deleting lock files..."
find user_data -name "*.lock" -delete 2>/dev/null || true
print_success "Deleted lock files"

# Delete any .tmp files
print_status "Deleting temporary files..."
find user_data -name "*.tmp" -delete 2>/dev/null || true
print_success "Deleted temporary files"

# Recreate logs directory
mkdir -p user_data/logs

print_header "Step 5: Verifying Clean State"

print_status "Checking for remaining SQLite databases..."
REMAINING=$(ls user_data/*.sqlite 2>/dev/null | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    print_success "✅ No database files remaining"
else
    print_warning "⚠️  Found $REMAINING database files still present"
fi

print_status "Checking logs directory..."
if [ -d "user_data/logs" ] && [ -z "$(ls -A user_data/logs)" ]; then
    print_success "✅ Logs directory is empty"
else
    print_warning "⚠️  Logs directory is not empty"
fi

print_header "Step 6: Starting Fresh"

print_status "Starting all containers with fresh state..."
if docker compose -f docker-compose-multi-strategies.yml up -d --force-recreate; then
    print_success "All containers started"

    print_status "Waiting for containers to initialize..."
    sleep 10

    # Show running containers
    print_status "Running containers:"
    docker compose -f docker-compose-multi-strategies.yml ps

    print_success "Containers are running"
else
    print_error "Failed to start containers"
    exit 1
fi

print_header "✅ Reset Complete!"

print_success "Trading data has been wiped and all bots restarted"
echo ""
print_status "Summary:"
echo "  ✅ All trade databases deleted"
echo "  ✅ All logs cleared"
echo "  ✅ Containers restarted with fresh state"
echo "  ✅ Backup saved to: $BACKUP_DIR"
echo ""
print_status "Next steps:"
echo "  1. Verify bots are running: docker compose -f docker-compose-multi-strategies.yml ps"
echo "  2. Check logs for errors: docker compose -f docker-compose-multi-strategies.yml logs -f"
echo "  3. Monitor first trades carefully"
echo "  4. Verify leverage settings in each bot"
echo ""
print_warning "Remember: All bots are starting fresh with NO trade history"
print_warning "First trades may behave differently than subsequent trades"
echo ""

# Show backup location
print_status "Backup location: $(pwd)/$BACKUP_DIR"
print_status "Backup size: $(du -sh $BACKUP_DIR | cut -f1)"
