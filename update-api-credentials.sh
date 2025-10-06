#!/bin/bash

# API Credentials Update Script for Multi-Strategy FreqTrade Setup
# This script helps update API credentials across all environment files

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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to update a single env file
update_env_file() {
    local env_file=$1
    local api_key=$2
    local api_secret=$3
    local dry_run=$4
    
    if [ ! -f "$env_file" ]; then
        print_error "Environment file not found: $env_file"
        return 1
    fi
    
    # Create backup
    cp "$env_file" "$env_file.backup"
    
    # Update API key
    sed -i.tmp "s/FREQTRADE__EXCHANGE__KEY=.*/FREQTRADE__EXCHANGE__KEY=$api_key/" "$env_file"
    
    # Update API secret
    sed -i.tmp "s/FREQTRADE__EXCHANGE__SECRET=.*/FREQTRADE__EXCHANGE__SECRET=$api_secret/" "$env_file"
    
    # Update dry run setting
    sed -i.tmp "s/FREQTRADE__DRY_RUN=.*/FREQTRADE__DRY_RUN=$dry_run/" "$env_file"
    
    # Remove temporary file
    rm -f "$env_file.tmp"
    
    print_success "Updated $env_file"
}

# Function to update all environment files
update_all_env_files() {
    local api_key=$1
    local api_secret=$2
    local dry_run=$3
    
    print_status "Updating all environment files..."
    
    # List of all environment files
    local env_files=(
        "env-files/nfi-x6.env"
        "env-files/bandtastic.env"
        "env-files/trendfollowing.env"
        "env-files/fvg.env"
        "env-files/powertower.env"
        "env-files/fastsupertrend.env"
    )
    
    for env_file in "${env_files[@]}"; do
        if [ -f "$env_file" ]; then
            update_env_file "$env_file" "$api_key" "$api_secret" "$dry_run"
        else
            print_warning "Environment file not found: $env_file"
        fi
    done
    
    print_success "All environment files updated!"
}

# Function to show current settings
show_current_settings() {
    print_status "Current API settings in environment files:"
    echo ""
    
    for env_file in env-files/*.env; do
        if [ -f "$env_file" ]; then
            echo "=== $(basename $env_file) ==="
            grep -E "FREQTRADE__EXCHANGE__(KEY|SECRET|NAME)" "$env_file" || true
            grep "FREQTRADE__DRY_RUN" "$env_file" || true
            echo ""
        fi
    done
}

# Function to validate API credentials
validate_credentials() {
    local api_key=$1
    local api_secret=$2
    
    if [ -z "$api_key" ] || [ "$api_key" == "Put_Your_Bybit_API_Key_Here" ]; then
        print_error "Please provide a valid API key"
        return 1
    fi
    
    if [ -z "$api_secret" ] || [ "$api_secret" == "Put_Your_Bybit_API_Secret_Here" ]; then
        print_error "Please provide a valid API secret"
        return 1
    fi
    
    if [ ${#api_key} -lt 10 ]; then
        print_error "API key seems too short"
        return 1
    fi
    
    if [ ${#api_secret} -lt 10 ]; then
        print_error "API secret seems too short"
        return 1
    fi
    
    return 0
}

# Function to restore backups
restore_backups() {
    print_status "Restoring environment files from backups..."
    
    for backup_file in env-files/*.env.backup; do
        if [ -f "$backup_file" ]; then
            original_file="${backup_file%.backup}"
            cp "$backup_file" "$original_file"
            print_success "Restored $(basename $original_file)"
        fi
    done
}

# Function to clean backups
clean_backups() {
    print_status "Cleaning backup files..."
    rm -f env-files/*.env.backup
    print_success "Backup files cleaned"
}

# Function to show help
show_help() {
    echo "API Credentials Update Script for Multi-Strategy FreqTrade"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  update                      Interactive update mode"
    echo "  update-all KEY SECRET [DRY] Update all env files with provided credentials"
    echo "  show                        Show current API settings"
    echo "  restore                     Restore from backup files"
    echo "  clean-backups              Remove backup files"
    echo "  help                        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 update"
    echo "  $0 update-all your_api_key your_api_secret true"
    echo "  $0 show"
    echo "  $0 restore"
}

# Interactive update mode
interactive_update() {
    echo ""
    echo "=== FreqTrade Multi-Strategy API Credentials Update ==="
    echo ""
    
    print_warning "This will update API credentials in ALL environment files!"
    echo ""
    
    # Get API key
    read -p "Enter your Bybit API Key: " api_key
    echo ""
    
    # Get API secret (hidden input)
    echo -n "Enter your Bybit API Secret: "
    read -s api_secret
    echo ""
    echo ""
    
    # Get dry run setting
    echo "Trading Mode:"
    echo "  1) Dry Run (Paper Trading) - RECOMMENDED for testing"
    echo "  2) Live Trading - Use real money"
    echo ""
    read -p "Select mode (1 or 2): " mode_choice
    
    if [ "$mode_choice" == "1" ]; then
        dry_run="true"
        print_warning "Selected: DRY RUN (Paper Trading)"
    elif [ "$mode_choice" == "2" ]; then
        dry_run="false"
        print_error "Selected: LIVE TRADING (Real Money!)"
        echo ""
        print_warning "WARNING: This will enable live trading with real money!"
        read -p "Are you absolutely sure? Type 'YES' to confirm: " confirm
        if [ "$confirm" != "YES" ]; then
            print_status "Cancelled by user"
            exit 0
        fi
    else
        print_error "Invalid choice. Using dry run mode."
        dry_run="true"
    fi
    
    echo ""
    
    # Validate credentials
    if ! validate_credentials "$api_key" "$api_secret"; then
        print_error "Invalid credentials provided"
        exit 1
    fi
    
    # Show summary
    echo "=== Summary ==="
    echo "API Key: ${api_key:0:8}...${api_key: -4}"
    echo "API Secret: ${api_secret:0:4}...${api_secret: -4}"
    echo "Dry Run: $dry_run"
    echo ""
    
    read -p "Proceed with update? (y/N): " proceed
    if [[ ! "$proceed" =~ ^[Yy]$ ]]; then
        print_status "Update cancelled"
        exit 0
    fi
    
    # Perform update
    update_all_env_files "$api_key" "$api_secret" "$dry_run"
    
    echo ""
    print_success "Update completed successfully!"
    print_status "Backup files created with .backup extension"
    print_status "You can now start your strategies with: ./deploy-multi-strategies.sh start"
}

# Main script logic
case "$1" in
    update)
        interactive_update
        ;;
    update-all)
        if [ -z "$2" ] || [ -z "$3" ]; then
            print_error "Usage: $0 update-all API_KEY API_SECRET [DRY_RUN]"
            exit 1
        fi
        
        api_key="$2"
        api_secret="$3"
        dry_run="${4:-true}"
        
        if ! validate_credentials "$api_key" "$api_secret"; then
            exit 1
        fi
        
        update_all_env_files "$api_key" "$api_secret" "$dry_run"
        ;;
    show)
        show_current_settings
        ;;
    restore)
        restore_backups
        ;;
    clean-backups)
        clean_backups
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
