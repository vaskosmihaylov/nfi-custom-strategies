# Multi-Strategy FreqTrade Setup Guide

This guide will help you set up multiple FreqTrade strategies with NGINX reverse proxy, allowing you to manage all strategies through a single FreqUI interface.

## 📋 Overview

The multi-strategy setup includes:
- **12 different trading strategies** running in separate Docker containers
- **NGINX reverse proxy** for unified access with proper path routing
- **Individual environment configurations** for each strategy
- **Single FreqUI interface** to manage all bots
- **Health monitoring** and deployment scripts
- **Proper CORS configuration** for cross-origin requests

## 🏗️ Architecture

```
Internet → NGINX (Port 80) → FreqTrade Strategies
                           ├── nfi-x7 (Port 8080)
                           ├── BandtasticFiboHyper (Port 8082)
                           ├── ElliotV5_SMA (Port 8091)
                           ├── BinClucMadV1 (Port 8092)
                           ├── NASOSv4 (Port 8093)
                           ├── RsiquiV5 (Port 8094)
                           ├── ElliotV5_SMA_Shorts (Port 8097)
                           ├── E0V1E (Port 8098)
                           ├── E0V1E_Shorts (Port 8099)
                           ├── EI4_t4c0s_V2_2 (Port 8100)
                           ├── EI4_t4c0s_V2_2_Shorts (Port 8101)
                           ├── ETCG (Port 8102)
                           └── ETCG_Shorts (Port 8103)
```

## 📁 Files Created

### Docker Configuration
- `docker-compose-multi-strategies.yml` - Multi-strategy Docker Compose file
- `deploy-multi-strategies.sh` - Deployment and management script

### NGINX Configuration
- `nginx-freqtrade-corrected.conf` - Corrected NGINX configuration with proper path routing
- `freqtrade-proxy-common.conf` - Reusable proxy headers

### Environment Files (in `env-files/`)
- `nfi-x7.env` - NostalgiaForInfinityX7 strategy
- `bandtastic.env` - BandtasticFiboHyper strategy (combined longs/shorts)
- `elliotv5_sma.env` - ElliotV5_SMA strategy (longs-only)
- `binclucmadv1.env` - BinClucMadV1 strategy
- `nasosv4.env` - NASOSv4 strategy
- `rsiquiv5.env` - RsiquiV5 strategy (supports both longs and shorts)
- `elliotv5_sma_shorts.env` - ElliotV5_SMA_Shorts strategy (shorts-only)
- `e0v1e.env` - E0V1E strategy (longs-only with 3x leverage)
- `e0v1e_shorts.env` - E0V1E_Shorts strategy (shorts-only with 3x leverage)
- `ei4_t4c0s_v2_2.env` - EI4_t4c0s_V2_2 strategy (longs with 3x leverage)
- `ei4_t4c0s_v2_2_shorts.env` - EI4_t4c0s_V2_2_Shorts strategy (shorts with 3x leverage)
- `etcg.env` - ETCG strategy (longs-only, multi-entry)
- `etcg_shorts.env` - ETCG_Shorts strategy (shorts-only, multi-entry)

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
./deploy-multi-strategies.sh start nfi-x7
./deploy-multi-strategies.sh start elliotv5_sma
```

### Step 3: Setup NGINX (requires sudo)

```bash
# Copy NGINX configuration and reload
sudo ./deploy-multi-strategies.sh setup-nginx
```

### Step 4: Access FreqUI

Open your browser and navigate to:
- **Main UI**: `http://freq.gaiaderma.com`
- **Health Check**: `http://freq.gaiaderma.com/health/nfi-x7`

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
| **nfi-x7** | `Vasko_NFI_X7` | `http://freq.gaiaderma.com/nfi-x7` | `nfi_x6_user` | `nfi_x6_secure_password` |
| **ElliotV5_SMA** | `Vasko_ElliotV5_SMA` | `http://freq.gaiaderma.com/elliotv5_sma` | `elliotv5_sma_user` | `elliotv5_sma_secure_password` |
| **BinClucMadV1** | `Vasko_BinClucMadV1` | `http://freq.gaiaderma.com/binclucmadv1` | `binclucmadv1_user` | `binclucmadv1_secure_password` |
| **NASOSv4** | `Vasko_NASOSv4` | `http://freq.gaiaderma.com/nasosv4` | `nasosv4_user` | `nasosv4_secure_password` |
| **RsiquiV5** | `Vasko_RsiquiV5` | `http://freq.gaiaderma.com/rsiquiv5` | `rsiquiv5_user` | `rsiquiv5_secure_password` |
| **ElliotV5_SMA_Shorts** | `Vasko_ElliotV5_SMA_Shorts` | `http://freq.gaiaderma.com/elliotv5_sma_shorts` | `elliotv5_sma_shorts_user` | `elliotv5_sma_shorts_secure_password` |
| **E0V1E** | `Vasko_E0V1E` | `http://freq.gaiaderma.com/e0v1e` | `e0v1e_user` | `e0v1e_secure_password` |
| **E0V1E_Shorts** | `Vasko_E0V1E_Shorts` | `http://freq.gaiaderma.com/e0v1e_shorts` | `e0v1e_shorts_user` | `e0v1e_shorts_secure_password` |
| **EI4_t4c0s_V2_2** | `Vasko_EI4_t4c0s_V2_2` | `http://freq.gaiaderma.com/ei4_t4c0s_v2_2` | `ei4_t4c0s_v2_2_user` | `ei4_t4c0s_v2_2_secure_password` |
| **EI4_t4c0s_V2_2_Shorts** | `Vasko_EI4_t4c0s_V2_2_Shorts` | `http://freq.gaiaderma.com/ei4_t4c0s_v2_2_shorts` | `ei4_t4c0s_v2_2_shorts_user` | `ei4_t4c0s_v2_2_shorts_secure_password` |
| **ETCG** | `Vasko_ETCG` | `http://freq.gaiaderma.com/etcg` | `etcg_user` | `etcg_secure_password` |
| **ETCG_Shorts** | `Vasko_ETCG_Shorts` | `http://freq.gaiaderma.com/etcg_shorts` | `etcg_shorts_user` | `etcg_shorts_secure_password` |

### ✅ URL Flow Example:
1. **FreqUI configured with**: `http://freq.gaiaderma.com/elliotv5_sma`
2. **FreqUI automatically appends**: `/api/v1/token/login`
3. **Final request**: `http://freq.gaiaderma.com/elliotv5_sma/api/v1/token/login`
4. **NGINX proxies to**: `http://127.0.0.1:8091/api/v1/token/login`

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

# Individual health checks (direct to containers)
curl http://127.0.0.1:8080/api/v1/ping  # nfi-x7
curl http://127.0.0.1:8082/api/v1/ping  # Bandtastic
curl http://127.0.0.1:8091/api/v1/ping  # ElliotV5_SMA
curl http://127.0.0.1:8092/api/v1/ping  # BinClucMadV1
curl http://127.0.0.1:8093/api/v1/ping  # NASOSv4
curl http://127.0.0.1:8094/api/v1/ping  # RsiquiV5
curl http://127.0.0.1:8097/api/v1/ping  # ElliotV5_SMA_Shorts
curl http://127.0.0.1:8098/api/v1/ping  # E0V1E
curl http://127.0.0.1:8099/api/v1/ping  # E0V1E_Shorts
curl http://127.0.0.1:8100/api/v1/ping  # EI4_t4c0s_V2_2
curl http://127.0.0.1:8101/api/v1/ping  # EI4_t4c0s_V2_2_Shorts
curl http://127.0.0.1:8102/api/v1/ping  # ETCG
curl http://127.0.0.1:8103/api/v1/ping  # ETCG_Shorts

# Test through NGINX
curl http://freq.gaiaderma.com/nfi-x7/api/v1/ping
curl http://freq.gaiaderma.com/elliotv5_sma/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e_shorts/api/v1/ping
curl http://freq.gaiaderma.com/ei4_t4c0s_v2_2/api/v1/ping
curl http://freq.gaiaderma.com/ei4_t4c0s_v2_2_shorts/api/v1/ping
curl http://freq.gaiaderma.com/etcg/api/v1/ping
curl http://freq.gaiaderma.com/etcg_shorts/api/v1/ping
```

### Log Management
```bash
# View all logs
./deploy-multi-strategies.sh logs

# View specific strategy logs
./deploy-multi-strategies.sh logs nfi-x7

# Live log following
docker compose -f docker-compose-multi-strategies.yml logs -f freqtrade-nfi-x7
```

### File-based Logs
Each strategy logs to separate files in `user_data/logs/`:
- `nfi-x7.log`
- `elliotv5_sma.log`
- etc.

## 🛠️ Configuration Details

### Shared Configuration
All strategies use the same base configuration (`configs/recommended_config.json`) but can be customized via environment variables.

### Port Allocation
- nfi-x7: 8080
- BandtasticFiboHyper: 8082
- ElliotV5_SMA: 8091
- BinClucMadV1: 8092
- NASOSv4: 8093
- RsiquiV5: 8094
- ElliotV5_SMA_Shorts: 8097
- E0V1E: 8098
- E0V1E_Shorts: 8099
- EI4_t4c0s_V2_2: 8100
- EI4_t4c0s_V2_2_Shorts: 8101
- ETCG: 8102
- ETCG_Shorts: 8103

### Database Separation
Each strategy uses its own SQLite database:
- `nfi-x7-tradesv3.sqlite`
- `bandtastic-tradesv3.sqlite`
- `elliotv5_sma-tradesv3.sqlite`
- `binclucmadv1-tradesv3.sqlite`
- `nasosv4-tradesv3.sqlite`
- `rsiquiv5-tradesv3.sqlite`
- `elliotv5_sma_shorts-tradesv3.sqlite`
- `e0v1e-tradesv3.sqlite`
- `e0v1e_shorts-tradesv3.sqlite`
- `ei4_t4c0s_v2_2-tradesv3.sqlite`
- `ei4_t4c0s_v2_2_shorts-tradesv3.sqlite`
- `etcg-tradesv3.sqlite`
- `etcg_shorts-tradesv3.sqlite`

### NGINX Path Routing
The NGINX configuration uses simple base paths without complex rewrites:
```nginx
# Strategy-specific paths (FreqUI appends /api/v1/* to these)
location /elliotv5_sma/ {
    proxy_pass http://127.0.0.1:8091/;
}
```

## 🚨 Troubleshooting

### Common Issues

1. **CORS/Double-Path Issues**

   **Problem**: Browser console shows `POST http://freq.gaiaderma.com/api/v1/elliotv5_sma/api/v1/token/login 405 (Method Not Allowed)`

   **Solution**: Remove `/api/v1/` from FreqUI bot URLs. Use `http://freq.gaiaderma.com/elliotv5_sma` instead of `http://freq.gaiaderma.com/api/v1/elliotv5_sma`

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
   curl http://freq.gaiaderma.com/nfi-x7/api/v1/ping
   
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
curl http://freq.gaiaderma.com/nfi-x7/api/v1/ping
curl http://freq.gaiaderma.com/bandtastic/api/v1/ping
curl http://freq.gaiaderma.com/elliotv5_sma/api/v1/ping
curl http://freq.gaiaderma.com/binclucmadv1/api/v1/ping
curl http://freq.gaiaderma.com/nasosv4/api/v1/ping
curl http://freq.gaiaderma.com/rsiquiv5/api/v1/ping
curl http://freq.gaiaderma.com/elliotv5_sma_shorts/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e_shorts/api/v1/ping
curl http://freq.gaiaderma.com/ei4_t4c0s_v2_2/api/v1/ping
curl http://freq.gaiaderma.com/ei4_t4c0s_v2_2_shorts/api/v1/ping
curl http://freq.gaiaderma.com/etcg/api/v1/ping
curl http://freq.gaiaderma.com/etcg_shorts/api/v1/ping

# Test health endpoints
curl http://freq.gaiaderma.com/health/nfi-x7
curl http://freq.gaiaderma.com/health/bandtastic
curl http://freq.gaiaderma.com/health/elliotv5_sma
curl http://freq.gaiaderma.com/health/e0v1e
curl http://freq.gaiaderma.com/health/e0v1e_shorts
curl http://freq.gaiaderma.com/health/ei4_t4c0s_v2_2
curl http://freq.gaiaderma.com/health/ei4_t4c0s_v2_2_shorts
curl http://freq.gaiaderma.com/health/etcg
curl http://freq.gaiaderma.com/health/etcg_shorts

# Check container status
./deploy-multi-strategies.sh status
./deploy-multi-strategies.sh health-check
```

All endpoints should return `{"status":"pong"}`.

---

**Happy Trading! 🚀**

For support, check the FreqTrade documentation: https://www.freqtrade.io/en/stable/

**Key Insight**: The most common issue is including `/api/v1/` in FreqUI bot URLs. FreqUI automatically appends API paths, so use base URLs like `http://freq.gaiaderma.com/elliotv5_sma` instead of `http://freq.gaiaderma.com/api/v1/elliotv5_sma`.
