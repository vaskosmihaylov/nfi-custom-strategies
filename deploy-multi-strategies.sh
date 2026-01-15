#!/bin/bash

# Multi-Strategy FreqTrade Deployment Script
# This script helps manage the multi-strategy FreqTrade setup

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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    
    if curl -s http://127.0.0.1:$port/api/v1/ping > /dev/null 2>&1; then
        print_success "$service_name (port $port) is running"
        return 0
    else
        print_error "$service_name (port $port) is not responding"
        return 1
    fi
}

# Function to display help
show_help() {
    echo "Multi-Strategy FreqTrade Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start [strategy]     Start all strategies or a specific strategy"
    echo "  stop [strategy]      Stop all strategies or a specific strategy"
    echo "  restart [strategy]   Restart all strategies or a specific strategy"
    echo "  status              Show status of all strategies"
    echo "  logs [strategy]     Show logs for all strategies or a specific strategy"
    echo "  setup-nginx         Copy NGINX configuration files"
    echo "  update-config       Update all environment files with real API credentials"
    echo "  health-check        Check health of all running strategies"
    echo "  help                Show this help message"
    echo ""
    echo "Available strategies:"
    echo "  nfi-x7, bandtastic, elliotv5_sma, elliotv5_sma_shorts,"
    echo "  binclucmadv1, nasosv4, martyema"
    echo ""
    echo "Examples:"
    echo "  \$0 start                    # Start all strategies"
    echo "  \$0 start nfi-x7            # Start only NFI-X7 strategy"
    echo "  \$0 status                  # Show status of all strategies"
    echo "  \$0 logs bandtastic         # Show logs for Bandtastic strategy"
    echo "  \$0 health-check            # Check health of all running strategies"
    echo "  \$0 setup-nginx             # Copy NGINX configuration"
    echo ""
    echo "FreqUI Bot URLs (use these in FreqUI):"
    echo "  NFI-X7:                http://freq.gaiaderma.com/nfi-x7"
    echo "  Bandtastic:            http://freq.gaiaderma.com/bandtastic"
    echo "  ElliotV5_SMA:          http://freq.gaiaderma.com/elliotv5_sma"
    echo "  ElliotV5_SMA_Shorts:   http://freq.gaiaderma.com/elliotv5_sma_shorts"
    echo "  BinClucMadV1:          http://freq.gaiaderma.com/binclucmadv1"
    echo "  NASOSv4:               http://freq.gaiaderma.com/nasosv4"
    echo "  MartyEMA:              http://freq.gaiaderma.com/martyema"
    echo "  (Note: Do NOT include /api/v1/ in URLs - FreqUI adds this automatically)"
}

# Function to start strategies
start_strategies() {
    local strategy=$1
    
    if [ -n "$strategy" ]; then
        print_status "Starting $strategy strategy..."
        docker compose -f docker-compose-multi-strategies.yml up -d freqtrade-$strategy
    else
        print_status "Starting all strategies..."
        docker compose -f docker-compose-multi-strategies.yml up -d
    fi
    
    print_success "Started strategies"
    sleep 5
    health_check
}

# Function to stop strategies
stop_strategies() {
    local strategy=$1
    
    if [ -n "$strategy" ]; then
        print_status "Stopping $strategy strategy..."
        docker compose -f docker-compose-multi-strategies.yml stop freqtrade-$strategy
    else
        print_status "Stopping all strategies..."
        docker compose -f docker-compose-multi-strategies.yml down
    fi
    
    print_success "Stopped strategies"
}

# Function to restart strategies
restart_strategies() {
    local strategy=$1
    
    if [ -n "$strategy" ]; then
        print_status "Restarting $strategy strategy..."
        docker compose -f docker-compose-multi-strategies.yml restart freqtrade-$strategy
    else
        print_status "Restarting all strategies..."
        docker compose -f docker-compose-multi-strategies.yml restart
    fi
    
    print_success "Restarted strategies"
    sleep 5
    health_check
}

# Function to show status
show_status() {
    print_status "Checking status of all strategies..."
    docker compose -f docker-compose-multi-strategies.yml ps
}

# Function to show logs
show_logs() {
    local strategy=$1
    
    if [ -n "$strategy" ]; then
        print_status "Showing logs for $strategy strategy..."
        docker compose -f docker-compose-multi-strategies.yml logs -f freqtrade-$strategy
    else
        print_status "Showing logs for all strategies..."
        docker compose -f docker-compose-multi-strategies.yml logs -f
    fi
}

# Function to setup NGINX
setup_nginx() {
    print_status "Setting up NGINX configuration..."
    
    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then
        print_warning "NGINX setup requires root privileges. Please run with sudo."
        return 1
    fi
    
    # Copy main NGINX configuration (use corrected version if available)
    if [ -f "nginx-freqtrade-corrected.conf" ]; then
        cp nginx-freqtrade-corrected.conf /etc/nginx/sites-available/freqtrade-multi
        print_success "Copied corrected NGINX configuration"
    elif [ -f "nginx-freqtrade-multi.conf" ]; then
        cp nginx-freqtrade-multi.conf /etc/nginx/sites-available/freqtrade-multi
        print_success "Copied main NGINX configuration"
    else
        print_error "No NGINX configuration file found (looking for nginx-freqtrade-corrected.conf or nginx-freqtrade-multi.conf)"
        return 1
    fi
    
    # Copy proxy headers configuration (use common version if available)
    if [ -f "freqtrade-proxy-common.conf" ]; then
        cp freqtrade-proxy-common.conf /etc/nginx/conf.d/
        print_success "Copied proxy common configuration"
    elif [ -f "freqtrade-proxy-headers.conf" ]; then
        cp freqtrade-proxy-headers.conf /etc/nginx/conf.d/
        print_success "Copied proxy headers configuration"
    else
        print_error "No proxy configuration file found (looking for freqtrade-proxy-common.conf or freqtrade-proxy-headers.conf)"
        return 1
    fi
    
    # Enable the site
    ln -sf /etc/nginx/sites-available/freqtrade-multi /etc/nginx/sites-enabled/
    print_success "Enabled FreqTrade multi-strategy site"
    
    # Test NGINX configuration
    if nginx -t; then
        print_success "NGINX configuration test passed"
        systemctl reload nginx
        print_success "NGINX reloaded"
    else
        print_error "NGINX configuration test failed"
        return 1
    fi
}

# Function to check health of all strategies
health_check() {
    print_status "Performing health check on all strategies..."
    
    local strategies=(
        "nfi-x7:8080"
        "bandtastic:8082"
        "elliotv5_sma:8091"
        "binclucmadv1:8092"
        "nasosv4:8093"
        "martyema:8094"
        "elliotv5_sma_shorts:8097"
    )
    
    local healthy=0
    local total=${#strategies[@]}
    
    for strategy_port in "${strategies[@]}"; do
        local strategy=${strategy_port%:*}
        local port=${strategy_port#*:}
        
        if check_service "$strategy" "$port"; then
            ((healthy++))
        fi
    done
    
    echo ""
    print_status "Health check summary: $healthy/$total strategies are healthy"
    
    if [ $healthy -eq $total ]; then
        print_success "All strategies are running properly!"
    else
        print_warning "Some strategies are not responding. Check the logs for details."
    fi
}

# Function to update configuration files
update_config() {
    print_warning "This function will help you update API credentials in environment files"
    print_warning "Make sure to update the following in each env file:"
    echo "  - FREQTRADE__EXCHANGE__KEY=Your_Real_API_Key"
    echo "  - FREQTRADE__EXCHANGE__SECRET=Your_Real_API_Secret"
    echo "  - FREQTRADE__DRY_RUN=false (for live trading)"
    echo ""
    print_status "CORS configuration should be (comma-separated format):"
    echo "  FREQTRADE__API_SERVER__CORS_ORIGINS=https://freq.gaiaderma.com,http://freq.gaiaderma.com"
    echo '  FREQTRADE__API_SERVER__FORWARDED_ALLOW_IPS="*"'
    echo ""
    print_status "Environment files are located in: env-files/"
    ls -la env-files/*.env 2>/dev/null || echo "No environment files found"
}

# Main script logic
case "$1" in
    start)
        start_strategies "$2"
        ;;
    stop)
        stop_strategies "$2"
        ;;
    restart)
        restart_strategies "$2"
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    setup-nginx)
        setup_nginx
        ;;
    health-check)
        health_check
        ;;
    update-config)
        update_config
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