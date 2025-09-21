# Multi-Strategy FreqTrade Setup Guide

This guide will help you set up multiple FreqTrade strategies with NGINX reverse proxy, allowing you to manage all strategies through a single FreqUI interface.

## 📋 Overview

The multi-strategy setup includes:
- **10 different trading strategies** running in separate Docker containers
- **NGINX reverse proxy** for unified access
- **Individual environment configurations** for each strategy
- **Single FreqUI interface** to manage all bots
- **Health monitoring** and deployment scripts

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
- `nginx-freqtrade-multi.conf` - Main NGINX configuration with path routing
- `freqtrade-proxy-headers.conf` - Reusable proxy headers

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
./deploy-multi-strategies.sh start quickadapter
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

Based on your screenshot, you'll add multiple bots to FreqUI. Each bot will have:

### Bot Configuration Format:
1. **Bot Name**: Descriptive name (e.g., "NFI-X6", "QuickAdapter")
2. **API URL**: Use strategy-specific endpoints:
   - NFI-X6: `http://freq.gaiaderma.com/api/v1/nfi-x6`
   - QuickAdapter: `http://freq.gaiaderma.com/api/v1/quickadapter`
   - BandtasticFibo: `http://freq.gaiaderma.com/api/v1/bandtastic`
   - TrendFollowing: `http://freq.gaiaderma.com/api/v1/trendfollowing`
   - Renko: `http://freq.gaiaderma.com/api/v1/renko`
   - FVG: `http://freq.gaiaderma.com/api/v1/fvg`
   - PowerTower: `http://freq.gaiaderma.com/api/v1/powertower`
   - FastSupertrend: `http://freq.gaiaderma.com/api/v1/fastsupertrend`
   - NoTankAI: `http://freq.gaiaderma.com/api/v1/notankai`
   - DTW: `http://freq.gaiaderma.com/api/v1/dtw`

3. **Username/Password**: Use the credentials from each strategy's env file:
   - Username: `{strategy}_user` (e.g., `nfi_x6_user`, `quickadapter_user`)
   - Password: `{strategy}_secure_password`

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

## 📊 Monitoring and Logs

### Health Monitoring
```bash
# Check all strategies health
./deploy-multi-strategies.sh health-check

# Individual health checks
curl http://127.0.0.1:8080/api/v1/ping  # NFI-X6
curl http://127.0.0.1:8081/api/v1/ping  # QuickAdapter
```

### Log Management
```bash
# View all logs
./deploy-multi-strategies.sh logs

# View specific strategy logs
./deploy-multi-strategies.sh logs nfi-x6

# Live log following
docker-compose -f docker-compose-multi-strategies.yml logs -f freqtrade-nfi-x6
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

## 🚨 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check if ports are in use
   netstat -tulpn | grep :808[0-9]
   
   # Stop conflicting services
   ./deploy-multi-strategies.sh stop
   ```

2. **NGINX Configuration Errors**
   ```bash
   # Test NGINX configuration
   sudo nginx -t
   
   # Check NGINX logs
   sudo tail -f /var/log/nginx/freqtrade_error.log
   ```

3. **Strategy Not Starting**
   ```bash
   # Check strategy logs
   ./deploy-multi-strategies.sh logs strategy-name
   
   # Check Docker container status
   docker-compose -f docker-compose-multi-strategies.yml ps
   ```

4. **API Connection Issues**
   ```bash
   # Test direct API access
   curl http://127.0.0.1:8080/api/v1/ping
   
   # Check strategy configuration
   cat env-files/strategy-name.env
   ```

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

---

**Happy Trading! 🚀**

For support, check the FreqTrade documentation: https://www.freqtrade.io/en/stable/