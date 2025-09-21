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
    echo "  nfi-x6, quickadapter, bandtastic, trendfollowing, renko,"
    echo "  fvg, powertower, fastsupertrend, notankai, dtw"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start all strategies"
    echo "  $0 start nfi-x6            # Start only NFI-X6 strategy"
    echo "  $0 status                  # Show status of all strategies"
    echo "  $0 logs quickadapter       # Show logs for QuickAdapter strategy"
    echo "  $0 setup-nginx             # Copy NGINX configuration"
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
    
    # Copy main NGINX configuration
    if [ -f "nginx-freqtrade-multi.conf" ]; then
        cp nginx-freqtrade-multi.conf /etc/nginx/sites-available/freqtrade-multi
        print_success "Copied main NGINX configuration"
    else
        print_error "nginx-freqtrade-multi.conf not found"
        return 1
    fi
    
    # Copy proxy headers configuration
    if [ -f "freqtrade-proxy-headers.conf" ]; then
        cp freqtrade-proxy-headers.conf /etc/nginx/conf.d/
        print_success "Copied proxy headers configuration"
    else
        print_error "freqtrade-proxy-headers.conf not found"
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
        "nfi-x6:8080"
        "quickadapter:8081"
        "bandtastic:8082"
        "trendfollowing:8083"
        "renko:8084"
        "fvg:8085"
        "powertower:8086"
        "fastsupertrend:8087"
        "notankai:8088"
        "dtw:8089"
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
    echo "  - FREQTRADE__EXCHANGE__KEY"
    echo "  - FREQTRADE__EXCHANGE__SECRET"
    echo "  - FREQTRADE__DRY_RUN (set to false for live trading)"
    echo ""
    print_status "Environment files are located in: env-files/"
    ls -la env-files/*.env
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