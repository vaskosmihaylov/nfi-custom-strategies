#!/bin/bash

###############################################################################
# NostalgiaForInfinity Upstream Update Script for Production Server
###############################################################################
#
# This script automatically updates your production fork with changes from
# the upstream NostalgiaForInfinity repository. It handles:
# - Adding/updating the upstream remote
# - Fetching and merging upstream changes
# - Detecting which strategies were updated
# - Restarting affected Docker containers
# - Telegram notifications
# - Comprehensive logging
#
# Designed for: Amazon EC2 production server
# Location: /opt/trading/nfi-custom-strategies
# Cron: Run every hour (0 * * * * /opt/trading/nfi-custom-strategies/tools/update_upstream_production.sh)
#
###############################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES FOR YOUR SETUP
# ============================================================================

# Repository configuration
REPO_PATH="/opt/trading/nfi-custom-strategies"
UPSTREAM_REPO="https://github.com/iterativv/NostalgiaForInfinity"
UPSTREAM_BRANCH="main"
UPSTREAM_REMOTE="upstream"

# Deployment script path (relative to REPO_PATH)
DEPLOY_SCRIPT="./deploy-multi-strategies.sh"

# Docker Compose file (if you want to restart entire stack instead of individual strategies)
DOCKER_COMPOSE_FILE=""  # Leave empty to use deploy script, or set to docker-compose file path

# Restart strategy - choose one:
# "auto" - Automatically restart strategies with updated files
# "all" - Restart all strategies on any update
# "manual" - Don't restart, just update files
RESTART_STRATEGY="auto"

# Merge strategy - choose one:
# "merge" - Standard git merge (recommended)
# "rebase" - Rebase your changes on top of upstream (use with caution)
MERGE_STRATEGY="merge"

# Conflict handling - choose one:
# "abort" - Abort update on conflicts, send notification
# "ours" - Keep our version on conflicts (use with caution)
# "theirs" - Use upstream version on conflicts (use with caution)
CONFLICT_STRATEGY="abort"

# Logging
LOG_FILE="/var/log/nfi-update.log"  # Update this path or use relative path
KEEP_LOGS_DAYS=30  # Keep logs for 30 days

# Telegram notifications (optional)
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=""  # Your bot token
TELEGRAM_CHAT_ID=""    # Your chat ID

# Send notifications for
NOTIFY_ON_SUCCESS=true   # Notify when update succeeds
NOTIFY_ON_NO_CHANGES=false  # Notify when no changes found
NOTIFY_ON_ERROR=true     # Notify on errors

# Strategy to container mapping (based on your deploy-multi-strategies.sh)
# Format: "strategy_file:container_name"
declare -A STRATEGY_CONTAINERS=(
    ["NostalgiaForInfinityX6.py"]="nfi-x6"
    ["NostalgiaForInfinityX7.py"]="nfi-x7"
    # Add more mappings as needed
    ["BandtasticFiboHyper.py"]="bandtastic"
    ["BandtasticFiboHyper_Combined.py"]="bandtastic"
    ["TrendFollowingStrategy.py"]="trendfollowing"
    ["FVGAdvancedStrategy_V2.py"]="fvg"
    ["PowerTower.py"]="powertower"
    ["FastSupertrend.py"]="fastsupertrend"
    ["MacheteV8b.py"]="machetev8b"
    ["ElliotV5_SMA_Shorts.py"]="elliotv5_sma"
    ["BinClucMadV1.py"]="binclucmadv1"
    ["NASOSv4.py"]="nasosv4"
    ["MartyEMA.py"]="martyema"
    ["Ichimoku_v2.py"]="ichimoku"
    ["BigWill.py"]="bigwill"
)

# ============================================================================
# Color Codes (disable if running in cron)
# ============================================================================

if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$message" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

# Send Telegram notification
send_telegram() {
    local message="$1"

    if [ "$TELEGRAM_ENABLED" = false ]; then
        return 0
    fi

    if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
        return 0
    fi

    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${message}" \
        -d "parse_mode=HTML" \
        -d "disable_web_page_preview=true" > /dev/null 2>&1
}

# Cleanup old logs
cleanup_old_logs() {
    if [ -f "$LOG_FILE" ]; then
        find "$(dirname "$LOG_FILE")" -name "$(basename "$LOG_FILE").*.gz" -mtime +${KEEP_LOGS_DAYS} -delete 2>/dev/null || true
    fi
}

# Rotate logs if too large (>10MB)
rotate_logs() {
    if [ -f "$LOG_FILE" ]; then
        local size=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
        if [ "$size" -gt 10485760 ]; then
            log_info "Rotating log file..."
            mv "$LOG_FILE" "$LOG_FILE.$(date +%Y%m%d_%H%M%S)"
            gzip "$LOG_FILE.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
        fi
    fi
}

# ============================================================================
# Git Functions
# ============================================================================

setup_upstream_remote() {
    log_info "Checking upstream remote..."

    if git remote | grep -q "^${UPSTREAM_REMOTE}$"; then
        log_info "Upstream remote '${UPSTREAM_REMOTE}' already exists"

        # Verify URL is correct
        local current_url=$(git remote get-url "$UPSTREAM_REMOTE")
        if [ "$current_url" != "$UPSTREAM_REPO" ]; then
            log_warn "Upstream URL mismatch. Updating..."
            git remote set-url "$UPSTREAM_REMOTE" "$UPSTREAM_REPO"
            log_success "Updated upstream remote URL"
        fi
    else
        log_info "Adding upstream remote..."
        git remote add "$UPSTREAM_REMOTE" "$UPSTREAM_REPO"
        log_success "Added upstream remote: $UPSTREAM_REPO"
    fi
}

fetch_upstream() {
    log_info "Fetching from upstream/${UPSTREAM_BRANCH}..."

    if git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Fetch completed successfully"
        return 0
    else
        log_error "Failed to fetch from upstream"
        return 1
    fi
}

get_changed_files() {
    # Get list of changed files between current HEAD and upstream
    git diff --name-only HEAD "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" 2>/dev/null || echo ""
}

merge_upstream() {
    local current_branch=$(git branch --show-current)
    log_info "Current branch: $current_branch"
    log_info "Merging upstream/${UPSTREAM_BRANCH}..."

    # Check if we're behind upstream
    local behind=$(git rev-list --count HEAD.."${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" 2>/dev/null || echo 0)

    if [ "$behind" -eq 0 ]; then
        log_info "Already up to date with upstream"
        return 2  # Special return code for "no changes"
    fi

    log_info "Behind upstream by $behind commits"

    # Get changed files before merge
    CHANGED_FILES=$(get_changed_files)

    if [ -z "$CHANGED_FILES" ]; then
        log_info "No file changes detected"
        # Still update the branch pointer
        git merge "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" --ff-only 2>&1 | tee -a "$LOG_FILE" || true
        return 2
    fi

    log_info "Changed files:"
    echo "$CHANGED_FILES" | while read -r file; do
        log_info "  - $file"
    done

    # Perform merge based on strategy
    if [ "$MERGE_STRATEGY" = "rebase" ]; then
        log_info "Using rebase strategy..."
        if git rebase "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" 2>&1 | tee -a "$LOG_FILE"; then
            log_success "Rebase completed successfully"
            return 0
        else
            handle_conflicts "rebase"
            return $?
        fi
    else
        log_info "Using merge strategy..."
        if git merge "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" -m "Merge upstream ${UPSTREAM_BRANCH}" 2>&1 | tee -a "$LOG_FILE"; then
            log_success "Merge completed successfully"
            return 0
        else
            handle_conflicts "merge"
            return $?
        fi
    fi
}

handle_conflicts() {
    local operation="$1"

    log_error "Conflicts detected during $operation"

    case "$CONFLICT_STRATEGY" in
        abort)
            log_warn "Aborting $operation due to conflicts"
            if [ "$operation" = "rebase" ]; then
                git rebase --abort 2>/dev/null || true
            else
                git merge --abort 2>/dev/null || true
            fi

            local conflict_msg="<b>🚨 NFI Update Failed - Conflicts Detected</b>%0A%0A"
            conflict_msg+="<b>Time:</b> $(date '+%Y-%m-%d %H:%M:%S')%0A"
            conflict_msg+="<b>Server:</b> $(hostname)%0A%0A"
            conflict_msg+="Conflicts occurred during upstream merge.%0A"
            conflict_msg+="Manual intervention required.%0A%0A"
            conflict_msg+="<b>Conflicting files:</b>%0A"
            git diff --name-only --diff-filter=U 2>/dev/null | while read -r file; do
                conflict_msg+="❌ $file%0A"
            done

            send_telegram "$conflict_msg"
            return 1
            ;;
        ours)
            log_warn "Resolving conflicts with 'ours' strategy"
            git checkout --ours . 2>/dev/null || true
            git add -A
            if [ "$operation" = "rebase" ]; then
                git rebase --continue 2>&1 | tee -a "$LOG_FILE" || true
            else
                git commit --no-edit 2>&1 | tee -a "$LOG_FILE" || true
            fi
            return 0
            ;;
        theirs)
            log_warn "Resolving conflicts with 'theirs' strategy"
            git checkout --theirs . 2>/dev/null || true
            git add -A
            if [ "$operation" = "rebase" ]; then
                git rebase --continue 2>&1 | tee -a "$LOG_FILE" || true
            else
                git commit --no-edit 2>&1 | tee -a "$LOG_FILE" || true
            fi
            return 0
            ;;
    esac
}

# ============================================================================
# Docker Functions
# ============================================================================

detect_affected_strategies() {
    local changed_files="$1"
    local affected_strategies=()

    # Check which strategies were affected
    while IFS= read -r file; do
        [ -z "$file" ] && continue

        # Extract filename
        local filename=$(basename "$file")

        # Check if this file maps to a container
        if [ -n "${STRATEGY_CONTAINERS[$filename]}" ]; then
            local container="${STRATEGY_CONTAINERS[$filename]}"
            # Avoid duplicates
            if [[ ! " ${affected_strategies[@]} " =~ " ${container} " ]]; then
                affected_strategies+=("$container")
            fi
        fi
    done <<< "$changed_files"

    echo "${affected_strategies[@]}"
}

restart_strategies() {
    local restart_list="$1"

    if [ -z "$restart_list" ]; then
        log_info "No strategies to restart"
        return 0
    fi

    log_info "Restarting affected strategies: $restart_list"

    # Check if deploy script exists
    if [ ! -f "$DEPLOY_SCRIPT" ]; then
        log_error "Deploy script not found: $DEPLOY_SCRIPT"
        return 1
    fi

    # Make sure it's executable
    chmod +x "$DEPLOY_SCRIPT" 2>/dev/null || true

    # Restart each strategy
    for strategy in $restart_list; do
        log_info "Restarting $strategy..."
        if "$DEPLOY_SCRIPT" restart "$strategy" 2>&1 | tee -a "$LOG_FILE"; then
            log_success "Restarted $strategy"
        else
            log_error "Failed to restart $strategy"
        fi
    done
}

restart_all_strategies() {
    log_info "Restarting all strategies..."

    if [ -n "$DOCKER_COMPOSE_FILE" ] && [ -f "$DOCKER_COMPOSE_FILE" ]; then
        log_info "Using docker-compose to restart all containers..."
        docker compose -f "$DOCKER_COMPOSE_FILE" restart 2>&1 | tee -a "$LOG_FILE"
    elif [ -f "$DEPLOY_SCRIPT" ]; then
        log_info "Using deploy script to restart all strategies..."
        "$DEPLOY_SCRIPT" restart 2>&1 | tee -a "$LOG_FILE"
    else
        log_error "No method available to restart strategies"
        return 1
    fi
}

# ============================================================================
# Main Function
# ============================================================================

main() {
    log_info "=========================================================================="
    log_info "NFI Upstream Update Script Starting"
    log_info "=========================================================================="

    # Cleanup and rotate logs
    cleanup_old_logs
    rotate_logs

    # Ensure log directory exists
    mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

    # Change to repository directory
    if [ ! -d "$REPO_PATH" ]; then
        log_error "Repository path does not exist: $REPO_PATH"
        send_telegram "🚨 NFI Update Error: Repository path not found"
        exit 1
    fi

    cd "$REPO_PATH" || exit 1
    log_info "Working directory: $(pwd)"

    # Verify it's a git repository
    if [ ! -d ".git" ]; then
        log_error "Not a git repository: $REPO_PATH"
        send_telegram "🚨 NFI Update Error: Not a git repository"
        exit 1
    fi

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        log_warn "Uncommitted changes detected. Stashing..."
        git stash push -m "Auto-stash before upstream update $(date +%Y%m%d_%H%M%S)" 2>&1 | tee -a "$LOG_FILE"
    fi

    # Setup upstream remote
    setup_upstream_remote || {
        log_error "Failed to setup upstream remote"
        send_telegram "🚨 NFI Update Error: Failed to setup upstream remote"
        exit 1
    }

    # Fetch from upstream
    fetch_upstream || {
        log_error "Failed to fetch from upstream"
        send_telegram "🚨 NFI Update Error: Failed to fetch from upstream"
        exit 1
    }

    # Merge upstream changes
    merge_upstream
    merge_result=$?

    if [ $merge_result -eq 2 ]; then
        # No changes
        log_success "Already up to date with upstream"
        if [ "$NOTIFY_ON_NO_CHANGES" = true ]; then
            send_telegram "ℹ️ NFI Update: No changes detected at $(date '+%H:%M')"
        fi
        log_info "=========================================================================="
        exit 0
    elif [ $merge_result -ne 0 ]; then
        # Merge failed
        log_error "Failed to merge upstream changes"
        exit 1
    fi

    # Update was successful
    log_success "Successfully merged upstream changes"

    # Detect affected strategies
    local affected_strategies=""
    if [ "$RESTART_STRATEGY" = "auto" ] && [ -n "$CHANGED_FILES" ]; then
        affected_strategies=$(detect_affected_strategies "$CHANGED_FILES")
        log_info "Affected strategies: ${affected_strategies:-none}"
    fi

    # Restart strategies based on configuration
    case "$RESTART_STRATEGY" in
        auto)
            if [ -n "$affected_strategies" ]; then
                restart_strategies "$affected_strategies"
            else
                log_info "No strategies need restarting"
            fi
            ;;
        all)
            restart_all_strategies
            ;;
        manual)
            log_info "Manual restart mode - skipping automatic restart"
            ;;
    esac

    # Send success notification
    if [ "$NOTIFY_ON_SUCCESS" = true ]; then
        local msg="<b>✅ NFI Updated Successfully</b>%0A%0A"
        msg+="<b>Time:</b> $(date '+%Y-%m-%d %H:%M:%S')%0A"
        msg+="<b>Server:</b> $(hostname)%0A%0A"

        if [ -n "$CHANGED_FILES" ]; then
            msg+="<b>📁 Updated files:</b>%0A"
            echo "$CHANGED_FILES" | head -10 | while read -r file; do
                msg+="  ✓ $file%0A"
            done

            local file_count=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
            if [ "$file_count" -gt 10 ]; then
                msg+="  ... and $((file_count - 10)) more%0A"
            fi
            msg+="%0A"
        fi

        if [ -n "$affected_strategies" ] && [ "$RESTART_STRATEGY" = "auto" ]; then
            msg+="<b>🔄 Restarted strategies:</b>%0A"
            for strategy in $affected_strategies; do
                msg+="  ♻️ $strategy%0A"
            done
        elif [ "$RESTART_STRATEGY" = "all" ]; then
            msg+="<b>🔄 Restarted:</b> All strategies%0A"
        fi

        send_telegram "$msg"
    fi

    log_success "Update process completed successfully"
    log_info "=========================================================================="
}

# ============================================================================
# Script Entry Point
# ============================================================================

# Run main function
main

exit 0
