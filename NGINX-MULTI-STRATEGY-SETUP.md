# Multi-Strategy FreqTrade NGINX Reverse Proxy Setup

This guide explains how to set up NGINX as a reverse proxy for multiple FreqTrade strategy containers using a single domain name.

## Overview

Instead of opening multiple firewall ports, this setup uses NGINX to route different URL paths to different FreqTrade containers:

- **Main Dashboard**: `http://freq.gaiaderma.com/` - Shows all available strategies
- **NFI Strategy**: `http://freq.gaiaderma.com/nfi/` → Container on port 8080
- **BandtasticFiboHyper**: `http://freq.gaiaderma.com/bandtastic/` → Container on port 8081
- **FVGAdvanced**: `http://freq.gaiaderma.com/fvgadvanced/` → Container on port 8083
- **PowerTower**: `http://freq.gaiaderma.com/powertower/` → Container on port 8084
- **TrendFollowing**: `http://freq.gaiaderma.com/trendfollowing/` → Container on port 8085
- **AdaptiveRenko**: `http://freq.gaiaderma.com/adaptiverenko/` → Container on port 8086
- **QuickAdapter**: `http://freq.gaiaderma.com/quickadapter/` → Container on port 8087

## Benefits

✅ **Single Domain**: Access all strategies through one domain  
✅ **Security**: Only port 80/443 needs to be open in firewall  
✅ **WebSocket Support**: Full FreqUI functionality including real-time updates  
✅ **Health Monitoring**: Built-in health checks for all strategies  
✅ **Rate Limiting**: Protection against abuse  
✅ **SSL Ready**: Easy HTTPS setup with Let's Encrypt  

## Prerequisites

1. **NGINX installed** on your EC2 instance
2. **Docker and Docker Compose** installed
3. **Domain name** pointing to your EC2 instance (freq.gaiaderma.com)
4. **FreqTrade strategies** located in `/Users/vasko.mihaylov/AI_Projects/NostalgiaForInfinity/user_data/strategies`

## Installation Steps

### 1. Start Your FreqTrade Containers

```bash
# Navigate to your project directory
cd /Users/vasko.mihaylov/AI_Projects/NostalgiaForInfinity

# Start all strategy containers
docker-compose -f docker-compose-multi-strategy.yml up -d

# Check that all containers are running
docker-compose -f docker-compose-multi-strategy.yml ps
```

### 2. Install NGINX Configuration

```bash
# Make the setup script executable (if not already done)
chmod +x setup-nginx-multi-strategy.sh

# Run the setup script with sudo
sudo ./setup-nginx-multi-strategy.sh
```

The script will:
- Install the NGINX configuration
- Enable the site
- Test the configuration
- Reload NGINX
- Show you the access URLs

### 3. Verify the Setup

1. **Check NGINX status**: `sudo systemctl status nginx`
2. **Test configuration**: `sudo nginx -t`
3. **View logs**: `sudo tail -f /var/log/nginx/freqtrade_multi_error.log`

### 4. Access Your Strategies

Visit `http://freq.gaiaderma.com/` to see the main dashboard with links to all strategies.

## AWS Security Group Configuration

You only need to open these ports in your EC2 security group:

| Port | Protocol | Source | Description |
|------|----------|---------|-------------|
| 22   | TCP      | Your IP | SSH access |
| 80   | TCP      | 0.0.0.0/0 | HTTP (NGINX) |
| 443  | TCP      | 0.0.0.0/0 | HTTPS (optional) |

**Remove** these ports if they were previously open:
- 8080-8087 (FreqTrade containers - now proxied through NGINX)

## Health Monitoring

The setup includes several health check endpoints:

```bash
# Check all strategies
curl http://freq.gaiaderma.com/health/

# Check individual strategy
curl http://freq.gaiaderma.com/health/nfi
curl http://freq.gaiaderma.com/health/bandtastic
# ... etc for other strategies
```

## SSL/HTTPS Setup (Recommended)

To enable HTTPS with Let's Encrypt:

```bash
# Install certbot
sudo apt update
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d freq.gaiaderma.com

# The certbot will automatically configure NGINX for HTTPS
```

After SSL setup, your strategies will be accessible via:
- `https://freq.gaiaderma.com/nfi/`
- `https://freq.gaiaderma.com/bandtastic/`
- etc.

## Troubleshooting

### Container Not Accessible

1. **Check container status**:
   ```bash
   docker-compose -f docker-compose-multi-strategy.yml ps
   ```

2. **Check container logs**:
   ```bash
   docker-compose -f docker-compose-multi-strategy.yml logs freqtrade-nfi
   ```

3. **Restart specific container**:
   ```bash
   docker-compose -f docker-compose-multi-strategy.yml restart freqtrade-nfi
   ```

### NGINX Issues

1. **Check NGINX status**: `sudo systemctl status nginx`
2. **Test configuration**: `sudo nginx -t`
3. **View error logs**: `sudo tail -f /var/log/nginx/freqtrade_multi_error.log`
4. **Restart NGINX**: `sudo systemctl restart nginx`

### Port Conflicts

If you see port binding errors:

1. **Check what's using the port**:
   ```bash
   sudo netstat -tulpn | grep :8080
   ```

2. **Kill processes if needed**:
   ```bash
   sudo kill -9 <PID>
   ```

3. **Restart containers**:
   ```bash
   docker-compose -f docker-compose-multi-strategy.yml down
   docker-compose -f docker-compose-multi-strategy.yml up -d
   ```

## Configuration Details

### NGINX Features Included

- **WebSocket Support**: For real-time FreqUI updates
- **Rate Limiting**: Protection against abuse (30 requests/minute general, 10/minute for API)
- **Security Headers**: XSS protection, frame options, content type sniffing protection
- **Health Checks**: Individual and aggregate health monitoring
- **Path Rewriting**: Clean URLs with proper path handling
- **Logging**: Separate access and error logs

### Container Ports

| Strategy | Container Name | Internal Port | External Port |
|----------|----------------|---------------|---------------|
| NFI | freqtrade-nfi | 8080 | 127.0.0.1:8080 |
| BandtasticFiboHyper | freqtrade-bandtastic | 8081 | 127.0.0.1:8081 |
| FVGAdvanced | freqtrade-fvgadvanced | 8083 | 127.0.0.1:8083 |
| PowerTower | freqtrade-powertower | 8084 | 127.0.0.1:8084 |
| TrendFollowing | freqtrade-trendfollowing | 8085 | 127.0.0.1:8085 |
| AdaptiveRenko | freqtrade-adaptiverenko | 8086 | 127.0.0.1:8086 |
| QuickAdapter | freqtrade-quickadapter | 8087 | 127.0.0.1:8087 |

## Maintenance

### Adding New Strategies

1. Add the new container to `docker-compose-multi-strategy.yml`
2. Add a new location block in the NGINX configuration
3. Update the main dashboard HTML to include the new strategy
4. Reload NGINX: `sudo systemctl reload nginx`

### Removing Strategies

1. Comment out the container in `docker-compose-multi-strategy.yml`
2. Comment out the corresponding location block in NGINX config
3. Update the main dashboard HTML
4. Reload NGINX: `sudo systemctl reload nginx`

### Log Rotation

NGINX logs are automatically rotated by the system. To manually rotate:

```bash
sudo logrotate -f /etc/logrotate.d/nginx
```

## Support

For issues with:
- **FreqTrade**: Check the [FreqTrade documentation](https://www.freqtrade.io/en/stable/)
- **NGINX**: Check the [NGINX documentation](https://nginx.org/en/docs/)
- **This setup**: Create an issue in the NostalgiaForInfinity repository