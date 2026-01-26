#!/bin/bash
# Simple Production Deployment Script
# Copy this to: /opt/trading/nfi-custom-strategies/pull-and-restart.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

cd /opt/trading/nfi-custom-strategies || {
    print_error "Could not cd to /opt/trading/nfi-custom-strategies"
    exit 1
}

print_status "Pulling latest changes from fork..."

if git pull origin main; then
    print_success "Pull successful"

    # Show NFI version
    if [ -f "NostalgiaForInfinityX7.py" ]; then
        VERSION=$(grep 'return "v' NostalgiaForInfinityX7.py 2>/dev/null | head -1 | sed 's/.*return "\(v[^"]*\)".*/\1/')
        CLASS=$(grep 'class Nostalgia.*X7' NostalgiaForInfinityX7.py 2>/dev/null | head -1 | sed 's/class \([^(]*\).*/\1/')
        if [ -n "$VERSION" ]; then
            print_status "NFI X7 Version: $VERSION"
        fi
        if [ -n "$CLASS" ]; then
            print_status "Class Name: $CLASS"
        fi
    fi

    # Restart all containers (use up -d --force-recreate to reload env files)
    print_status "Restarting docker containers (reloading configs)..."
    if docker compose -f docker-compose-multi-strategies.yml up -d --force-recreate; then
        print_success "Containers restarted successfully"

        # Wait for containers to start
        print_status "Waiting for containers to start..."
        sleep 10

        # Check for errors in logs
        ERROR_COUNT=$(docker compose -f docker-compose-multi-strategies.yml logs --tail=50 2>&1 | grep -i "impossible to load\|error.*strategy" | wc -l)

        if [ "$ERROR_COUNT" -gt 0 ]; then
            print_warning "Found errors in container logs:"
            docker compose -f docker-compose-multi-strategies.yml logs --tail=20
            print_warning "Check logs with: docker compose -f docker-compose-multi-strategies.yml logs"
        else
            print_success "No errors detected in container logs"
        fi

        # Show running containers
        print_status "Running containers:"
        docker compose -f docker-compose-multi-strategies.yml ps

        print_success "Deployment complete"
    else
        print_error "Container restart failed"
        exit 1
    fi
else
    print_error "Git pull failed"
    exit 1
fi
