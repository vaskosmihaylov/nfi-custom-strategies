#!/bin/bash

# Multi-Strategy FreqTrade NGINX Setup Script
# This script sets up NGINX reverse proxy for multiple FreqTrade strategies

set -e

echo "🚀 Setting up NGINX reverse proxy for multi-strategy FreqTrade..."

# Configuration variables
NGINX_CONF_FILE="nginx-freqtrade-multi-strategy.conf"
NGINX_SITES_AVAILABLE="/etc/nginx/sites-available"
NGINX_SITES_ENABLED="/etc/nginx/sites-enabled"
SITE_NAME="freqtrade-multi-strategy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run this script as root or with sudo"
    exit 1
fi

# Check if NGINX is installed
if ! command -v nginx &> /dev/null; then
    print_error "NGINX is not installed. Please install NGINX first:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install nginx"
    echo "  Amazon Linux: sudo yum install nginx"
    exit 1
fi

# Check if the configuration file exists
if [ ! -f "$NGINX_CONF_FILE" ]; then
    print_error "NGINX configuration file '$NGINX_CONF_FILE' not found!"
    print_error "Please ensure the file exists in the current directory."
    exit 1
fi

print_status "Found NGINX configuration file: $NGINX_CONF_FILE"

# Backup existing configuration if it exists
if [ -f "$NGINX_SITES_AVAILABLE/$SITE_NAME" ]; then
    print_warning "Existing configuration found. Creating backup..."
    cp "$NGINX_SITES_AVAILABLE/$SITE_NAME" "$NGINX_SITES_AVAILABLE/$SITE_NAME.backup.$(date +%Y%m%d_%H%M%S)"
    print_status "Backup created successfully"
fi

# Copy the new configuration
print_status "Installing new NGINX configuration..."
cp "$NGINX_CONF_FILE" "$NGINX_SITES_AVAILABLE/$SITE_NAME"

# Remove default site if it exists and is enabled
if [ -L "$NGINX_SITES_ENABLED/default" ]; then
    print_status "Removing default NGINX site..."
    rm "$NGINX_SITES_ENABLED/default"
fi

# Enable the new site
if [ -L "$NGINX_SITES_ENABLED/$SITE_NAME" ]; then
    print_status "Site already enabled, updating symlink..."
    rm "$NGINX_SITES_ENABLED/$SITE_NAME"
fi

ln -s "$NGINX_SITES_AVAILABLE/$SITE_NAME" "$NGINX_SITES_ENABLED/$SITE_NAME"
print_status "Site enabled successfully"

# Test NGINX configuration
print_status "Testing NGINX configuration..."
if nginx -t; then
    print_status "NGINX configuration test passed ✅"
else
    print_error "NGINX configuration test failed ❌"
    print_error "Please check the configuration and try again."
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p /var/log/nginx
chown -R www-data:www-data /var/log/nginx 2>/dev/null || chown -R nginx:nginx /var/log/nginx 2>/dev/null || true

# Reload NGINX
print_status "Reloading NGINX..."
if systemctl reload nginx; then
    print_status "NGINX reloaded successfully ✅"
else
    print_error "Failed to reload NGINX ❌"
    exit 1
fi

# Check NGINX status
if systemctl is-active --quiet nginx; then
    print_status "NGINX is running ✅"
else
    print_warning "NGINX is not running. Starting NGINX..."
    systemctl start nginx
    if systemctl is-active --quiet nginx; then
        print_status "NGINX started successfully ✅"
    else
        print_error "Failed to start NGINX ❌"
        exit 1
    fi
fi

echo ""
echo "🎉 Multi-strategy NGINX setup completed successfully!"
echo ""
echo "📋 Configuration Summary:"
echo "  - Main dashboard: http://freq.gaiaderma.com/"
echo "  - NFI Strategy: http://freq.gaiaderma.com/nfi/"
echo "  - BandtasticFiboHyper: http://freq.gaiaderma.com/bandtastic/"
echo "  - FVGAdvanced: http://freq.gaiaderma.com/fvgadvanced/"
echo "  - PowerTower: http://freq.gaiaderma.com/powertower/"
echo "  - TrendFollowing: http://freq.gaiaderma.com/trendfollowing/"
echo "  - AdaptiveRenko: http://freq.gaiaderma.com/adaptiverenko/"
echo "  - QuickAdapter: http://freq.gaiaderma.com/quickadapter/"
echo ""
echo "🔍 Health Checks:"
echo "  - All strategies: http://freq.gaiaderma.com/health/"
echo "  - Individual: http://freq.gaiaderma.com/health/[strategy-name]"
echo ""
echo "📝 Next Steps:"
echo "  1. Ensure all FreqTrade containers are running:"
echo "     docker-compose -f docker-compose-multi-strategy.yml up -d"
echo ""
echo "  2. Check that all containers are healthy:"
echo "     docker-compose -f docker-compose-multi-strategy.yml ps"
echo ""
echo "  3. Test the setup by visiting: http://freq.gaiaderma.com/"
echo ""
echo "  4. For HTTPS setup, consider using Let's Encrypt:"
echo "     sudo certbot --nginx -d freq.gaiaderma.com"
echo ""

# Optional: Show running containers
if command -v docker-compose &> /dev/null; then
    echo "🐳 Current Docker Containers Status:"
    if [ -f "docker-compose-multi-strategy.yml" ]; then
        docker-compose -f docker-compose-multi-strategy.yml ps 2>/dev/null || echo "  Run: docker-compose -f docker-compose-multi-strategy.yml up -d"
    else
        echo "  docker-compose-multi-strategy.yml not found in current directory"
    fi
fi

echo ""
echo "✅ Setup complete! Your multi-strategy FreqTrade setup is ready to use."