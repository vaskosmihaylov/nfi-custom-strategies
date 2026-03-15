# Multi-Strategy FreqTrade Setup Guide

This guide will help you set up multiple FreqTrade strategies with NGINX reverse proxy, allowing you to manage all strategies through a single FreqUI interface.

## Overview

The multi-strategy setup includes:
- **17 active trading strategies** running in separate Docker containers
- **NGINX reverse proxy** for unified access with proper path routing
- **Individual environment configurations** for each strategy
- **Single FreqUI interface** to manage all bots
- **Health monitoring** and deployment scripts
- **Proper CORS configuration** for cross-origin requests

## Architecture

```
Internet → NGINX (Port 80) → FreqTrade Strategies
                           ├── nfi-x7 (Port 8080)
                           ├── E0V1E (Port 8098)
                           ├── E0V1E_Shorts (Port 8099)
                           ├── Auto_EI_t4c0s (Port 8100)
                           ├── ETCG_Shorts (Port 8103)
                           ├── KamaFama (Port 8091)
                           ├── ZaratustraDCA2_06 (Port 8119)
                           ├── BollingerBounce (Port 8124)
                           ├── BollingerBounce_Shorts (Port 8125)
                           ├── KeltnerBounce (Port 8126)
                           ├── KeltnerBounce_Shorts (Port 8127)
                           ├── UltraSmartStrategy_NoStoploss_v2 (Port 8128)
                           ├── Lmao (Port 8129)
                           ├── GKD_FisherTransformV4_ML (Port 8130)
                           ├── ATGDFV2 file strategy / AlexBandSniper (Port 8131)
                           └── NOTankAi_15_Cleaned_v2 (Port 8132)
```

## Files

### Docker Configuration
- `docker-compose-multi-strategies.yml` - Multi-strategy Docker Compose file
- `deploy-multi-strategies.sh` - Deployment and management script

### NGINX Configuration
- `nginx-freqtrade-multi.conf` - NGINX configuration with proper path routing
- `freqtrade-proxy-headers.conf` - Reusable proxy headers

### Environment Files (in `env-files/`)
- `nfi-x7.env` - NostalgiaForInfinityX7 strategy
- `e0v1e.env` - E0V1E strategy (longs-only with 3x leverage)
- `e0v1e_shorts.env` - E0V1E_Shorts strategy (shorts-only with 3x leverage)
- `auto_ei_t4c0s.env` - Auto_EI_t4c0s strategy (longs, weighted EWO scoring)
- `etcg_shorts.env` - ETCG_Shorts strategy (shorts-only, multi-entry)
- `kamafama.env` - KamaFama optimized long strategy (3x leverage, KAMA/FAMA mean-reversion)
- `zaratustra.env` - ZaratustraDCA2_06 strategy (longs + shorts with DCA and protection logic)
- `bollingerbounce.env` - BollingerBounce strategy (longs with 3x leverage)
- `bollingerbounce_shorts.env` - BollingerBounce_Shorts strategy (shorts-only with 3x leverage)
- `keltnerbounce.env` - KeltnerBounce strategy (longs with 3x leverage)
- `keltnerbounce_shorts.env` - KeltnerBounce_Shorts strategy (shorts-only with 3x leverage)
- `ultrasmart_nostop_v2.env` - UltraSmartStrategy_NoStoploss_v2 strategy (long-only Lmao family strategy)
- `lmao.env` - Lmao strategy (long-only Lmao family strategy)
- `gkd_transformv55_ml.env` - GKD_FisherTransformV4_ML strategy (ML-enhanced futures strategy)
- `atgdfv2.env` - ATGDFV2 file strategy using runtime class `AlexBandSniper`
- `notankai.env` - NOTankAi_15_Cleaned_v2 strategy (selected NOTankAi dry-run candidate)

## Quick Start

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
./deploy-multi-strategies.sh start auto_ei_t4c0s
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

## Management Commands

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

## Adding Bots to FreqUI

### IMPORTANT: Corrected URL Format

FreqUI expects **base URLs** and automatically appends API paths. Do **NOT** include `/api/v1/` in the middle of URLs.

### Bot Configuration:

| Strategy | Bot Name | API URL | Username | Password |
|----------|----------|---------|----------|----------|
| **nfi-x7** | `Vasko_NFI_X7` | `http://freq.gaiaderma.com/nfi-x7` | `nfi_x6_user` | `nfi_x6_secure_password` |
| **E0V1E** | `Vasko_E0V1E` | `http://freq.gaiaderma.com/e0v1e` | `e0v1e_user` | `e0v1e_secure_password` |
| **E0V1E_Shorts** | `Vasko_E0V1E_Shorts` | `http://freq.gaiaderma.com/e0v1e_shorts` | `e0v1e_shorts_user` | `e0v1e_shorts_secure_password` |
| **Auto_EI_t4c0s** | `Vasko_Auto_EI_t4c0s` | `http://freq.gaiaderma.com/auto_ei_t4c0s` | `auto_ei_t4c0s_user` | `auto_ei_t4c0s_secure_password` |
| **ETCG_Shorts** | `Vasko_ETCG_Shorts` | `http://freq.gaiaderma.com/etcg_shorts` | `etcg_shorts_user` | `etcg_shorts_secure_password` |
| **KamaFama** | `Vasko_KamaFama` | `http://freq.gaiaderma.com/kamafama` | `kamafama_user` | `kamafama_secure_password` |
| **ZaratustraDCA2_06** | `Vasko_ZaratustraDCA2_06` | `http://freq.gaiaderma.com/zaratustra` | `zaratustra_user` | `zaratustra_secure_password` |
| **BollingerBounce** | `Vasko_BollingerBounce` | `http://freq.gaiaderma.com/bollingerbounce` | `bollingerbounce_user` | `bollingerbounce_secure_password` |
| **BollingerBounce_Shorts** | `Vasko_BollingerBounce_Shorts` | `http://freq.gaiaderma.com/bollingerbounce_shorts` | `bollingerbounce_shorts_user` | `bollingerbounce_shorts_secure_password` |
| **KeltnerBounce** | `Vasko_KeltnerBounce` | `http://freq.gaiaderma.com/keltnerbounce` | `keltnerbounce_user` | `keltnerbounce_secure_password` |
| **KeltnerBounce_Shorts** | `Vasko_KeltnerBounce_Shorts` | `http://freq.gaiaderma.com/keltnerbounce_shorts` | `keltnerbounce_shorts_user` | `keltnerbounce_shorts_secure_password` |
| **UltraSmartStrategy_NoStoploss_v2** | `Vasko_UltraSmart_NoStop_v2` | `http://freq.gaiaderma.com/ultrasmart_nostop_v2` | `ultrasmart_nostop_v2_user` | `ultrasmart_nostop_v2_secure_password` |
| **Lmao** | `Vasko_Lmao` | `http://freq.gaiaderma.com/lmao` | `lmao_user` | `lmao_secure_password` |
| **GKD_FisherTransformV4_ML** | `Vasko_GKD_FisherTransformV4_ML` | `http://freq.gaiaderma.com/gkd_transformv55_ml` | `gkd_transformv55_ml_user` | `gkd_transformv55_ml_secure_password` |
| **ATGDFV2 / AlexBandSniper** | `Vasko_ATGDFV2` | `http://freq.gaiaderma.com/atgdfv2` | `atgdfv2_user` | `atgdfv2_secure_password` |
| **NOTankAi_15_Cleaned_v2** | `Vasko_NOTankAi` | `http://freq.gaiaderma.com/notankai` | `notankai_user` | `notankai_secure_password` |

### URL Flow Example:
1. **FreqUI configured with**: `http://freq.gaiaderma.com/auto_ei_t4c0s`
2. **FreqUI automatically appends**: `/api/v1/token/login`
3. **Final request**: `http://freq.gaiaderma.com/auto_ei_t4c0s/api/v1/token/login`
4. **NGINX proxies to**: `http://127.0.0.1:8100/api/v1/token/login`

## Security Considerations

### Environment Files
- Store API credentials securely
- Use strong, unique passwords for each strategy
- Consider using Docker secrets for production

### NGINX Security
- The configuration includes security headers
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

## Monitoring and Logs

### Health Monitoring
```bash
# Check all strategies health
./deploy-multi-strategies.sh health-check

# Individual health checks (direct to containers)
curl http://127.0.0.1:8080/api/v1/ping  # nfi-x7
curl http://127.0.0.1:8098/api/v1/ping  # E0V1E
curl http://127.0.0.1:8099/api/v1/ping  # E0V1E_Shorts
curl http://127.0.0.1:8100/api/v1/ping  # Auto_EI_t4c0s
curl http://127.0.0.1:8103/api/v1/ping  # ETCG_Shorts
curl http://127.0.0.1:8091/api/v1/ping  # KamaFama
curl http://127.0.0.1:8119/api/v1/ping  # ZaratustraDCA2_06
curl http://127.0.0.1:8124/api/v1/ping  # BollingerBounce
curl http://127.0.0.1:8125/api/v1/ping  # BollingerBounce_Shorts
curl http://127.0.0.1:8126/api/v1/ping  # KeltnerBounce
curl http://127.0.0.1:8127/api/v1/ping  # KeltnerBounce_Shorts
curl http://127.0.0.1:8128/api/v1/ping  # UltraSmartStrategy_NoStoploss_v2
curl http://127.0.0.1:8129/api/v1/ping  # Lmao
curl http://127.0.0.1:8130/api/v1/ping  # GKD_FisherTransformV4_ML
curl http://127.0.0.1:8131/api/v1/ping  # ATGDFV2 / AlexBandSniper
curl http://127.0.0.1:8132/api/v1/ping  # NOTankAi_15_Cleaned_v2

# Test through NGINX
curl http://freq.gaiaderma.com/nfi-x7/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e_shorts/api/v1/ping
curl http://freq.gaiaderma.com/auto_ei_t4c0s/api/v1/ping
curl http://freq.gaiaderma.com/etcg_shorts/api/v1/ping
curl http://freq.gaiaderma.com/kamafama/api/v1/ping
curl http://freq.gaiaderma.com/zaratustra/api/v1/ping
curl http://freq.gaiaderma.com/bollingerbounce/api/v1/ping
curl http://freq.gaiaderma.com/bollingerbounce_shorts/api/v1/ping
curl http://freq.gaiaderma.com/keltnerbounce/api/v1/ping
curl http://freq.gaiaderma.com/keltnerbounce_shorts/api/v1/ping
curl http://freq.gaiaderma.com/ultrasmart_nostop_v2/api/v1/ping
curl http://freq.gaiaderma.com/lmao/api/v1/ping
curl http://freq.gaiaderma.com/gkd_transformv55_ml/api/v1/ping
curl http://freq.gaiaderma.com/atgdfv2/api/v1/ping
curl http://freq.gaiaderma.com/notankai/api/v1/ping
curl http://freq.gaiaderma.com/gkd_transformv55_ml/api/v1/ping
curl http://freq.gaiaderma.com/atgdfv2/api/v1/ping
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
- `e0v1e.log`, `e0v1e_shorts.log`
- `auto_ei_t4c0s.log`
- `etcg_shorts.log`
- `kamafama.log`
- etc.

## Configuration Details

### Shared Configuration
All strategies use the same base configuration (`user_data/strategies/config.json`) but can be customized via environment variables.

### Port Allocation

| Port | Strategy | Direction | Leverage |
|------|----------|-----------|----------|
| 8080 | nfi-x7 | Longs | - |
| 8098 | E0V1E | Longs | 3x |
| 8099 | E0V1E_Shorts | Shorts | 3x |
| 8100 | Auto_EI_t4c0s | Longs | - |
| 8103 | ETCG_Shorts | Shorts | 3x |
| 8091 | KamaFama | Longs | 3x |
| 8119 | ZaratustraDCA2_06 | Longs + Shorts | Config-defined |
| 8124 | BollingerBounce | Longs | 3x |
| 8125 | BollingerBounce_Shorts | Shorts | 3x |
| 8126 | KeltnerBounce | Longs | 3x |
| 8127 | KeltnerBounce_Shorts | Shorts | 3x |
| 8128 | UltraSmartStrategy_NoStoploss_v2 | Longs | Config-defined |
| 8129 | Lmao | Longs | Config-defined |
| 8130 | GKD_FisherTransformV4_ML | Longs + Shorts | Strategy-defined |
| 8131 | ATGDFV2 / AlexBandSniper | Longs + Shorts | 7x leverage in strategy |

**Freed ports** (available for future strategies): 8097, 8104, 8112, 8118, 8132+

### Database Separation
Each strategy uses its own SQLite database:
- `nfi-x7-tradesv3.sqlite`
- `e0v1e-tradesv3.sqlite`
- `e0v1e_shorts-tradesv3.sqlite`
- `auto_ei_t4c0s-tradesv3.sqlite`
- `etcg_shorts-tradesv3.sqlite`
- `kamafama-tradesv3.sqlite`
- `zaratustra-tradesv3.sqlite`
- `bollingerbounce-tradesv3.sqlite`
- `bollingerbounce_shorts-tradesv3.sqlite`
- `keltnerbounce-tradesv3.sqlite`
- `keltnerbounce_shorts-tradesv3.sqlite`
- `ultrasmart_nostop_v2-tradesv3.sqlite`
- `lmao-tradesv3.sqlite`
- `gkd_transformv55_ml-tradesv3.sqlite`
- `atgdfv2-tradesv3.sqlite`

### NGINX Path Routing
The NGINX configuration uses simple base paths without complex rewrites:
```nginx
# Strategy-specific paths (FreqUI appends /api/v1/* to these)
location /auto_ei_t4c0s/ {
    proxy_pass http://127.0.0.1:8100/;
}
```

## Troubleshooting

### Common Issues

1. **CORS/Double-Path Issues**

   **Problem**: Browser console shows `POST http://freq.gaiaderma.com/api/v1/strategy/api/v1/token/login 405`

   **Solution**: Remove `/api/v1/` from FreqUI bot URLs. Use `http://freq.gaiaderma.com/auto_ei_t4c0s` instead of `http://freq.gaiaderma.com/api/v1/auto_ei_t4c0s`

2. **Port Conflicts**
   ```bash
   # Check if ports are in use
   netstat -tulpn | grep :8[01][0-9][0-9]

   # Stop and remove old containers holding ports
   docker stop <old-container-name>
   docker rm <old-container-name>

   # Then restart
   ./deploy-multi-strategies.sh start
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
2. **Verify CORS headers**: Use browser Developer Tools > Network tab
3. **Check NGINX configuration**: Ensure proxy headers are set correctly
4. **Restart containers**: After changing environment files

## Updating Strategies

To add new strategies or modify existing ones:

1. **Add new strategy environment file** in `env-files/`
2. **Update `docker-compose-multi-strategies.yml`** with new service
3. **Add NGINX location block** in `nginx-freqtrade-multi.conf`
4. **Update `deploy-multi-strategies.sh`** health check array and help text
5. **Restart services**

## Validation Tests

After setup, verify everything works:

```bash
# Quick health check for all strategies
./deploy-multi-strategies.sh health-check

# Check container status
./deploy-multi-strategies.sh status
```

All endpoints should return `{"status":"pong"}`.

---

For support, check the FreqTrade documentation: https://www.freqtrade.io/en/stable/

**Key Insight**: The most common issue is including `/api/v1/` in FreqUI bot URLs. FreqUI automatically appends API paths, so use base URLs like `http://freq.gaiaderma.com/auto_ei_t4c0s` instead of `http://freq.gaiaderma.com/api/v1/auto_ei_t4c0s`.

**Last Updated**: March 4, 2026

## Recent Changes (March 4, 2026)
- **Removed**: SimpleRSI (port 8124) and SimpleRSI_Shorts (port 8125)
- **Added**: BollingerBounce (longs, port 8124) and BollingerBounce_Shorts (shorts, port 8125), both 3x leverage
- **Added**: KeltnerBounce (longs, port 8126) and KeltnerBounce_Shorts (shorts, port 8127), both 3x leverage

## Recent Changes (March 11, 2026)
- **Added**: ZaratustraDCA2_06 (mixed long/short, port 8119) with `/zaratustra` reverse-proxy routing

## Recent Changes (February 9, 2026)
- **Removed**: BinClucMadV1 strategy (port 8092)
