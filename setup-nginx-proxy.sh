#!/bin/bash

# Configuration script for NGINX Freqtrade reverse proxy setup
# Run this script on your EC2 instance

echo "🔧 Configuring NGINX for Freqtrade reverse proxy..."

# Get EC2 public IP automatically
EC2_PUBLIC_domain="freq.gaiaderma.com"
echo "🌐 Detected EC2 public domain: $EC2_PUBLIC_domain"

# Copy the configuration file
echo "📁 Installing NGINX configuration..."
sudo cp nginx-freqtrade.conf /etc/nginx/sites-available/freqtrade

# Replace placeholder with actual IP
sudo sed -i "s/YOUR_EC2_PUBLIC_domain/$EC2_PUBLIC_domain/g" /etc/nginx/sites-available/freqtrade

# Enable the site
sudo ln -s /etc/nginx/sites-available/freqtrade /etc/nginx/sites-enabled/

# Remove default site if it exists
sudo rm -f /etc/nginx/sites-enabled/default

# Test NGINX configuration
echo "🧪 Testing NGINX configuration..."
if sudo nginx -t; then
    echo "✅ NGINX configuration test passed!"
    
    # Reload NGINX
    echo "🔄 Reloading NGINX..."
    sudo systemctl reload nginx
    echo "✅ NGINX reloaded successfully!"
    
    echo ""
    echo "🎉 NGINX reverse proxy setup completed!"
    echo "📊 Access your Freqtrade UI at: http://$EC2_PUBLIC_domain/"
    echo "🔍 Health check endpoint: http://$EC2_PUBLIC_domain/health"
    echo ""
    echo "🔒 Security Notes:"
    echo "   - Consider setting up SSL/HTTPS for production use"
    echo "   - Ensure your EC2 security group allows port 80"
    echo "   - Strong passwords are configured in your Freqtrade config"
    echo ""
else
    echo "❌ NGINX configuration test failed!"
    echo "Please check the configuration and try again."
    exit 1
fi

# Check if Freqtrade container is running
echo "🐳 Checking Freqtrade container status..."
if docker ps | grep -q "NFI_Vasko_bybit_futures"; then
    echo "✅ Freqtrade container is running"
    
    # Test the proxy connection
    echo "🔗 Testing reverse proxy connection..."
    if curl -s "http://localhost:8989/api/v1/ping" > /dev/null; then
        echo "✅ Freqtrade API is responding on port 8989"
    else
        echo "⚠️  Warning: Freqtrade API not responding on port 8989"
        echo "   Make sure your Freqtrade container has API enabled"
    fi
else
    echo "⚠️  Warning: Freqtrade container not found"
    echo "   Make sure your Docker container is running"
fi

echo ""
echo "📋 Next Steps:"
echo "   1. Update your EC2 Security Group to allow port 80"
echo "   2. Test access from your browser"
echo "   3. Consider setting up SSL with Let's Encrypt (certbot)"
echo "   4. Monitor logs: sudo tail -f /var/log/nginx/freqtrade_*.log"