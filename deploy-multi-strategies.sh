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
    echo "  nfi-x6, bandtastic, trendfollowing, fvg, powertower,"
    echo "  fastsupertrend, machetev8b,"
    echo "  elliotv5_sma, binclucmadv1, nasosv4, martyema,"
    echo "  ichimoku, bigwill"
    echo ""
    echo "Examples:"
    echo "  \$0 start                    # Start all strategies"
    echo "  \$0 start nfi-x6            # Start only NFI-X6 strategy"
    echo "  \$0 status                  # Show status of all strategies"
    echo "  \$0 logs bandtastic         # Show logs for Bandtastic strategy"
