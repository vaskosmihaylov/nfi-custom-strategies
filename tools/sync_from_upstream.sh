#!/bin/bash

###############################################################################
# Sync Mac from Upstream NFI - Manual Quick Sync
###############################################################################
#
# This script pulls the latest changes from upstream NFI and pushes to your fork.
# Use this when you want to get upstream changes immediately without waiting
# for the daily sync or production server sync.
#
# Usage:
#   ./sync_from_upstream.sh           # Pull from upstream, push to fork
#   ./sync_from_upstream.sh --dry-run # Show what would be done
#
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
UPSTREAM_REMOTE="origin"                    # iterativv/NostalgiaForInfinity
UPSTREAM_BRANCH="main"
FORK_REMOTE="nfi-custom-strategies"         # Your fork
FORK_BRANCH="main"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo ""
log_info "=========================================================================="
log_info "Sync Mac from Upstream NFI"
log_info "=========================================================================="
echo ""

# Check for uncommitted changes
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    log_error "You have uncommitted changes. Please commit or stash them first."
    echo ""
    git status --short
    echo ""
    exit 1
fi

# Get current branch
current_branch=$(git branch --show-current)
log_info "Current branch: $current_branch"

# Fetch from upstream
log_info "Fetching from upstream (${UPSTREAM_REMOTE})..."
git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"

# Check how many commits behind
behind=$(git rev-list --count HEAD.."${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" 2>/dev/null || echo 0)
ahead=$(git rev-list --count "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"..HEAD 2>/dev/null || echo 0)

log_info "Behind upstream: $behind commits"
log_info "Ahead of upstream: $ahead commits"

if [ "$behind" -eq 0 ]; then
    log_success "Already up to date with upstream!"
    exit 0
fi

# Show what's new
echo ""
log_info "New commits from upstream:"
echo ""
git log --oneline --graph --decorate HEAD.."${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" | head -10
echo ""

# Show changed files
log_info "Changed files:"
git diff --name-status HEAD.."${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" | head -20
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warn "DRY RUN - No changes made"
    log_info "Run without --dry-run to apply changes"
    exit 0
fi

# Ask for confirmation
read -p "$(echo -e ${CYAN}Pull these $behind commits from upstream? [Y/n]: ${NC})" response
response=${response:-y}
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    log_warn "Sync cancelled"
    exit 0
fi

# Pull from upstream
echo ""
log_info "Pulling from ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}..."
if git pull "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"; then
    log_success "Pulled $behind commits successfully"
else
    log_error "Pull failed - please resolve conflicts manually"
    exit 1
fi

# Ask about pushing to fork
echo ""
read -p "$(echo -e ${CYAN}Push changes to your fork (${FORK_REMOTE})? [Y/n]: ${NC})" response
response=${response:-y}
if [[ "$response" =~ ^[Yy]$ ]]; then
    log_info "Pushing to ${FORK_REMOTE}/${FORK_BRANCH}..."
    if git push "$FORK_REMOTE" "$current_branch"; then
        log_success "Pushed to fork successfully"
    else
        log_warn "Push to fork failed - you can push manually later"
    fi
fi

echo ""
log_info "=========================================================================="
log_success "Sync complete!"
log_info "=========================================================================="
echo ""
log_info "Your Mac is now up to date with upstream NFI"
if [[ "$response" =~ ^[Yy]$ ]]; then
    log_info "Your fork has been updated too"
fi
echo ""
