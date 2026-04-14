#!/bin/bash

# Multi-Strategy FreqTrade Deployment Script
# This script helps manage the multi-strategy FreqTrade setup

set -e

MULTI_COMPOSE_FILE="docker-compose-multi-strategies.yml"

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

get_compose_file_for_strategy() {
    echo "$MULTI_COMPOSE_FILE"
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
    echo "  nfi-x7, fastsupertrend_rsi_70, fastsupertrend_quick3, e0v1e_binance, e0v1e_binance_shorts, binhv27,"
    echo "  auto_ei_t4c0s, fibonacciematrend,"
    echo "  kamafama,"
    echo "  zaratustra,"
    echo "  bollingerbounce, bollingerbounce_shorts,"
    echo "  keltnerbounce, keltnerbounce_shorts,"
    echo "  ultrasmart_nostop_v2, fenix,"
    echo "  mtfscalper,"
    echo "  alexbandsniper_v10ai, triplesupertrendadxrsi, best5m,"
    echo "  cluc7werk, edtma, donchian_adx_chop, combinedbinhandclucv8, combinedbinhandclucv8xh"
    echo ""
    echo "Examples:"
    echo "  \$0 start                    # Start all strategies"
    echo "  \$0 start nfi-x7            # Start only NFI-X7 strategy"
    echo "  \$0 status                  # Show status of all strategies"
    echo "  \$0 logs fastsupertrend_rsi_70  # Show logs for FastSupertrend_optim3_rsi_70"
    echo "  \$0 health-check            # Check health of all running strategies"
    echo "  \$0 setup-nginx             # Copy NGINX configuration"
    echo ""
    echo "FreqUI Bot URLs (use these in FreqUI):"
    echo "  NFI-X7:                http://freq.gaiaderma.com/nfi-x7"
    echo "  FastSupertrend_rsi_70: http://freq.gaiaderma.com/fastsupertrend_rsi_70"
    echo "  FastSupertrend_quick3: http://freq.gaiaderma.com/fastsupertrend_quick3"
    echo "  E0V1E_Binance:         http://freq.gaiaderma.com/e0v1e_binance"
    echo "  E0V1E_Binance_Shorts:  http://freq.gaiaderma.com/e0v1e_binance_shorts"
    echo "  BinHV27:               http://freq.gaiaderma.com/binhv27"
    echo "  Auto_EI_t4c0s:         http://freq.gaiaderma.com/auto_ei_t4c0s"
    echo "  FibonacciEMATrend:     http://freq.gaiaderma.com/fibonacciematrend"
    echo "  KamaFama:              http://freq.gaiaderma.com/kamafama"
    echo "  ZaratustraDCA2_06:     http://freq.gaiaderma.com/zaratustra"
    echo "  BollingerBounce:       http://freq.gaiaderma.com/bollingerbounce"
    echo "  BollingerBounce_Shorts: http://freq.gaiaderma.com/bollingerbounce_shorts"
    echo "  KeltnerBounce:         http://freq.gaiaderma.com/keltnerbounce"
    echo "  KeltnerBounce_Shorts:  http://freq.gaiaderma.com/keltnerbounce_shorts"
    echo "  UltraSmart_NoStop_v2:  http://freq.gaiaderma.com/ultrasmart_nostop_v2"
    echo "  FenixTopProfit:        http://freq.gaiaderma.com/fenix"
    echo "  MtfScalper:            http://freq.gaiaderma.com/mtfscalper"
    echo "  AlexBandSniperV10AI:    http://freq.gaiaderma.com/alexbandsniper_v10ai"
    echo "  TripleSuperTrendADXRSI: http://freq.gaiaderma.com/triplesupertrendadxrsi"
    echo "  Best5m:                http://freq.gaiaderma.com/best5m"
    echo "  Cluc7werk:             http://freq.gaiaderma.com/cluc7werk"
    echo "  EDTMA:                 http://freq.gaiaderma.com/edtma"
    echo "  Donchian_ADX_CHOP:     http://freq.gaiaderma.com/donchian_adx_chop"
    echo "  CombinedBinHAndClucV8: http://freq.gaiaderma.com/combinedbinhandclucv8"
    echo "  CombinedBinHAndClucV8XH: http://freq.gaiaderma.com/combinedbinhandclucv8xh"
    echo "  (Note: Do NOT include /api/v1/ in URLs - FreqUI adds this automatically)"
}

# Function to start strategies
start_strategies() {
    local strategy=$1
    
    if [ -n "$strategy" ]; then
        print_status "Starting $strategy strategy..."
        docker compose -f "$(get_compose_file_for_strategy "$strategy")" up -d freqtrade-$strategy
    else
        print_status "Starting all strategies..."
        docker compose -f "$MULTI_COMPOSE_FILE" up -d
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
        docker compose -f "$(get_compose_file_for_strategy "$strategy")" stop freqtrade-$strategy
    else
        print_status "Stopping all strategies..."
        docker compose -f "$MULTI_COMPOSE_FILE" down
    fi
    
    print_success "Stopped strategies"
}

# Function to restart strategies
restart_strategies() {
    local strategy=$1

    if [ -n "$strategy" ]; then
        print_status "Restarting $strategy strategy (reloading env vars)..."
        # Use up -d --force-recreate to reload env files
        # docker compose restart does NOT reload env files!
        docker compose -f "$(get_compose_file_for_strategy "$strategy")" up -d --force-recreate freqtrade-$strategy
    else
        print_status "Restarting all strategies (reloading env vars)..."
        docker compose -f "$MULTI_COMPOSE_FILE" up -d --force-recreate
    fi

    print_success "Restarted strategies"
    sleep 5
    health_check
}

# Function to show status
show_status() {
    print_status "Checking status of all strategies..."
    docker compose -f "$MULTI_COMPOSE_FILE" ps
}

# Function to show logs
show_logs() {
    local strategy=$1
    
    if [ -n "$strategy" ]; then
        print_status "Showing logs for $strategy strategy..."
        docker compose -f "$(get_compose_file_for_strategy "$strategy")" logs -f freqtrade-$strategy
    else
        print_status "Showing logs for all strategies..."
        docker compose -f "$MULTI_COMPOSE_FILE" logs -f
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
        "fastsupertrend_rsi_70:8098"
        "fastsupertrend_quick3:8099"
        "e0v1e_binance:8114"
        "e0v1e_binance_shorts:8115"
        "binhv27:8092"
        "auto_ei_t4c0s:8100"
        "fibonacciematrend:8103"
        "kamafama:8091"
        "zaratustra:8119"
        "bollingerbounce:8124"
        "bollingerbounce_shorts:8125"
        "keltnerbounce:8126"
        "keltnerbounce_shorts:8127"
        "ultrasmart_nostop_v2:8128"
        "fenix:8129"
        "mtfscalper:8131"
        "alexbandsniper_v10ai:8132"
        "triplesupertrendadxrsi:8134"
        "best5m:8135"
        "cluc7werk:8136"
        "edtma:8137"
        "donchian_adx_chop:8138"
        "combinedbinhandclucv8:8139"
        "combinedbinhandclucv8xh:8140"
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
