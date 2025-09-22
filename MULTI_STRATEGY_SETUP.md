# Multi-Strategy FreqTrade Setup Guide

This guide will help you set up multiple FreqTrade strategies with NGINX reverse proxy, allowing you to manage all strategies through a single FreqUI interface.

## 📋 Overview

The multi-strategy setup includes:
- **10 different trading strategies** running in separate Docker containers
- **NGINX reverse proxy** for unified access with proper path routing
- **Individual environment configurations** for each strategy
- **Single FreqUI interface** to manage all bots
- **Health monitoring** and deployment scripts
- **Proper CORS configuration** for cross-origin requests

## 🏗️ Architecture

```
Internet → NGINX (Port 80) → FreqTrade Strategies
                           ├── NFI-X6 (Port 8080)
                           ├── QuickAdapter (Port 8081)
                           ├── BandtasticFiboHyper (Port 8082)
                           ├── TrendFollowing (Port 8083)
                           ├── Renko (Port 8084)
                           ├── FVG (Port 8085)
                           ├── PowerTower (Port 8086)
                           ├── FastSupertrend (Port 8087)
                           ├── NoTankAI (Port 8088)
                           └── DTW (Port 8089)
```

## 📁 Files Created

### Docker Configuration
- `docker-compose-multi-strategies.yml` - Multi-strategy Docker Compose file
- `deploy-multi-strategies.sh` - Deployment and management script

### NGINX Configuration
- `nginx-freqtrade-corrected.conf` - Corrected NGINX configuration with proper path routing
- `freqtrade-proxy-common.conf` - Reusable proxy headers

### Environment Files (in `env-files/`)
- `nfi-x6.env` - NostalgiaForInfinityX6 strategy
- `quickadapter.env` - QuickAdapter strategy
- `bandtastic.env` - BandtasticFiboHyper strategy
- `trendfollowing.env` - TrendFollowing strategy
- `renko.env` - Renko strategy
- `fvg.env` - FVG Advanced Strategy
- `powertower.env` - PowerTower strategy
- `fastsupertrend.env` - FastSupertrend strategy
- `notankai.env` - NoTankAI strategy
- `dtw.env` - DTW strategy

## 🚀 Quick Start

### Step 1: Update API Credentials

Before starting, update the API credentials in each environment file:

```bash
# Edit each env file and update:
# - FREQTRADE__EXCHANGE__KEY=Your_Real_API_Key
# - FREQTRADE__EXCHANGE__SECRET=Your_Real_API_Secret
# - FREQTRADE__DRY_RUN=false (for live trading)

# Use the helper script to see all env files:
./deploy-multi-strategies.sh update-config
```

### Step 2: Start All Strategies

```bash
# Start all strategies
./deploy-multi-strategies.sh start

# Or start individual strategies
./deploy-multi-strategies.sh start nfi-x6
./deploy-multi-strategies.sh start bandtastic
```

### Step 3: Setup NGINX (requires sudo)

```bash
# Copy NGINX configuration and reload
sudo ./deploy-multi-strategies.sh setup-nginx
```

### Step 4: Access FreqUI

Open your browser and navigate to:
- **Main UI**: `http://freq.gaiaderma.com`
- **Health Check**: `http://freq.gaiaderma.com/health/nfi-x6`

## 🎮 Management Commands

The `deploy-multi-strategies.sh` script provides comprehensive management:

```bash
# Start/Stop/Restart
./deploy-multi-strategies.sh start [strategy]
./deploy-multi-strategies.sh stop [strategy]
./deploy-multi-strategies.sh restart [strategy]

# Monitor
./deploy-multi-strategies.sh status
./deploy-multi-strategies.sh health-check
./deploy-multi-strategies.sh logs [strategy]

# Setup
./deploy-multi-strategies.sh setup-nginx
./deploy-multi-strategies.sh update-config
```

## 🔧 Adding Bots to FreqUI

### ⚠️ IMPORTANT: Corrected URL Format

FreqUI expects **base URLs** and automatically appends API paths. Do **NOT** include `/api/v1/` in the middle of URLs.

### Bot Configuration (Corrected URLs):

| Strategy | Bot Name | API URL | Username | Password |
|----------|----------|---------|----------|----------|
| **NFI-X6** | `Vasko_NFI_X6` | `http://freq.gaiaderma.com/nfi-x6` | `nfi_x6_user` | `nfi_x6_secure_password` |
| **Bandtastic** | `Vasko_Bandtastic` | `http://freq.gaiaderma.com/bandtastic` | `bandtastic_user` | `bandtastic_secure_password` |
| **QuickAdapter** | `Vasko_QuickAdapter` | `http://freq.gaiaderma.com/quickadapter` | `quickadapter_user` | `quickadapter_secure_password` |
| **TrendFollowing** | `Vasko_TrendFollowing` | `http://freq.gaiaderma.com/trendfollowing` | `trendfollowing_user` | `trendfollowing_secure_password` |
| **Renko** | `Vasko_Renko` | `http://freq.gaiaderma.com/renko` | `renko_user` | `renko_secure_password` |
| **FVG** | `Vasko_FVG` | `http://freq.gaiaderma.com/fvg` | `fvg_user` | `fvg_secure_password` |
| **PowerTower** | `Vasko_PowerTower` | `http://freq.gaiaderma.com/powertower` | `powertower_user` | `powertower_secure_password` |
| **FastSupertrend** | `Vasko_FastSupertrend` | `http://freq.gaiaderma.com/fastsupertrend` | `fastsupertrend_user` | `fastsupertrend_secure_password` |
| **NoTankAI** | `Vasko_NoTankAI` | `http://freq.gaiaderma.com/notankai` | `notankai_user` | `notankai_secure_password` |
| **DTW** | `Vasko_DTW` | `http://freq.gaiaderma.com/dtw` | `dtw_user` | `dtw_secure_password` |

### ✅ URL Flow Example:
1. **FreqUI configured with**: `http://freq.gaiaderma.com/bandtastic`
2. **FreqUI automatically appends**: `/api/v1/token/login`
3. **Final request**: `http://freq.gaiaderma.com/bandtastic/api/v1/token/login`
4. **NGINX proxies to**: `http://127.0.0.1:8082/api/v1/token/login`

## 🔒 Security Considerations

### Environment Files
- Store API credentials securely
- Use strong, unique passwords for each strategy
- Consider using Docker secrets for production

### NGINX Security
- The configuration includes rate limiting
- Security headers are applied
- Consider adding SSL/TLS for production use

### Network Security
- All strategies bind to `127.0.0.1` (localhost only)
- Only NGINX is exposed to external traffic

### CORS Configuration
Each environment file includes proper CORS configuration:
```bash
# Correct format for environment variables (comma-separated)
FREQTRADE__API_SERVER__CORS_ORIGINS=https://freq.gaiaderma.com,http://freq.gaiaderma.com
FREQTRADE__API_SERVER__FORWARDED_ALLOW_IPS="*"
```

## 📊 Monitoring and Logs

### Health Monitoring
```bash
# Check all strategies health
./deploy-multi-strategies.sh health-check

# Individual health checks
curl http://127.0.0.1:8080/api/v1/ping  # NFI-X6
curl http://127.0.0.1:8082/api/v1/ping  # Bandtastic

# Test through NGINX
curl http://freq.gaiaderma.com/nfi-x6/api/v1/ping
curl http://freq.gaiaderma.com/bandtastic/api/v1/ping
```

### Log Management
```bash
# View all logs
./deploy-multi-strategies.sh logs

# View specific strategy logs
./deploy-multi-strategies.sh logs nfi-x6

# Live log following
docker compose -f docker-compose-multi-strategies.yml logs -f freqtrade-nfi-x6
```

### File-based Logs
Each strategy logs to separate files in `user_data/logs/`:
- `nfi-x6.log`
- `quickadapter.log`
- `bandtastic.log`
- etc.

## 🛠️ Configuration Details

### Shared Configuration
All strategies use the same base configuration (`configs/recommended_config.json`) but can be customized via environment variables.

### Port Allocation
- NFI-X6: 8080
- QuickAdapter: 8081
- BandtasticFiboHyper: 8082
- TrendFollowing: 8083
- Renko: 8084
- FVG: 8085
- PowerTower: 8086
- FastSupertrend: 8087
- NoTankAI: 8088
- DTW: 8089

### Database Separation
Each strategy uses its own SQLite database:
- `nfi-x6-tradesv3.sqlite`
- `quickadapter-tradesv3.sqlite`
- etc.

### NGINX Path Routing
The NGINX configuration uses simple base paths without complex rewrites:
```nginx
# Strategy-specific paths (FreqUI appends /api/v1/* to these)
location /bandtastic/ {
    proxy_pass http://127.0.0.1:8082/;
}
```

## 🚨 Troubleshooting

### Common Issues

1. **CORS/Double-Path Issues**
   
   **Problem**: Browser console shows `POST http://freq.gaiaderma.com/api/v1/bandtastic/api/v1/token/login 405 (Method Not Allowed)`
   
   **Solution**: Remove `/api/v1/` from FreqUI bot URLs. Use `http://freq.gaiaderma.com/bandtastic` instead of `http://freq.gaiaderma.com/api/v1/bandtastic`

2. **Port Conflicts**
   ```bash
   # Check if ports are in use
   netstat -tulpn | grep :808[0-9]
   
   # Stop conflicting services
   ./deploy-multi-strategies.sh stop
   ```

3. **NGINX Configuration Errors**
   ```bash
   # Test NGINX configuration
   sudo nginx -t
   
   # Check NGINX logs
   sudo tail -f /var/log/nginx/freqtrade_error.log
   ```

4. **Strategy Not Starting**
   ```bash
   # Check strategy logs
   ./deploy-multi-strategies.sh logs strategy-name
   
   # Check Docker container status
   docker compose -f docker-compose-multi-strategies.yml ps
   ```

5. **API Connection Issues**
   ```bash
   # Test direct API access
   curl http://127.0.0.1:8080/api/v1/ping
   
   # Test through NGINX
   curl http://freq.gaiaderma.com/nfi-x6/api/v1/ping
   
   # Check strategy configuration
   cat env-files/strategy-name.env
   ```

### CORS Troubleshooting
If CORS issues persist:
1. **Check environment variable format**: Must be comma-separated, not JSON array
2. **Verify CORS headers**: Use browser Developer Tools → Network tab
3. **Check NGINX configuration**: Ensure proxy headers are set correctly
4. **Restart containers**: After changing environment files

### Performance Optimization

1. **Resource Limits**: Consider adding resource limits to Docker containers
2. **Data Sharing**: All strategies share the same `user_data/data` directory for efficiency
3. **Log Rotation**: Implement log rotation for production use

## 🔄 Updating Strategies

To add new strategies or modify existing ones:

1. **Add new strategy environment file**
2. **Update `docker-compose-multi-strategies.yml`**
3. **Add NGINX location block**
4. **Update deployment script**
5. **Restart services**

## 📈 Next Steps

1. **SSL/TLS Setup**: Configure HTTPS for production
2. **Monitoring**: Add Prometheus/Grafana monitoring
3. **Backup Strategy**: Implement database backup automation
4. **Scaling**: Consider Kubernetes for larger deployments

## 🔍 Validation Tests

After setup, verify everything works:

```bash
# Test all endpoints
curl http://freq.gaiaderma.com/nfi-x6/api/v1/ping
curl http://freq.gaiaderma.com/bandtastic/api/v1/ping
curl http://freq.gaiaderma.com/quickadapter/api/v1/ping

# Test health endpoints
curl http://freq.gaiaderma.com/health/nfi-x6
curl http://freq.gaiaderma.com/health/bandtastic

# Check container status
./deploy-multi-strategies.sh status
./deploy-multi-strategies.sh health-check
```

All endpoints should return `{"status":"pong"}`.

---

**Happy Trading! 🚀**

For support, check the FreqTrade documentation: https://www.freqtrade.io/en/stable/

**Key Insight**: The most common issue is including `/api/v1/` in FreqUI bot URLs. FreqUI automatically appends API paths, so use base URLs like `http://freq.gaiaderma.com/bandtastic` instead of `http://freq.gaiaderma.com/api/v1/bandtastic`.