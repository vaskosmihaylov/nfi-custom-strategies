# Multi-Strategy FreqTrade Setup Guide

This guide will help you set up multiple FreqTrade strategies with NGINX reverse proxy, allowing you to manage all strategies through a single FreqUI interface.

## Overview

The multi-strategy setup includes:
- **25 active trading strategies** running in separate Docker containers
- **NGINX reverse proxy** for unified access with proper path routing
- **Individual environment configurations** for each strategy
- **Single FreqUI interface** to manage all bots
- **Health monitoring** and deployment scripts
- **Proper CORS configuration** for cross-origin requests

## Architecture

```
Internet → NGINX (Port 80) → FreqTrade Strategies
                           ├── nfi-x7 (Port 8080)
                           ├── FalconTrader (Port 8092)
                           ├── FalconTrader_Short (Port 8094)
                           ├── E0V1E (Port 8098)
                           ├── E0V1E_Shorts (Port 8099)
                           ├── Auto_EI_t4c0s (Port 8100)
                           ├── Auto_EI_t4c0s_Shorts (Port 8101)
                           ├── ETCG (Port 8102)
                           ├── ETCG_Shorts (Port 8103)
                           ├── ClucHAnix_hhll (Port 8106)
                           ├── ClucHAnix_hhll_Shorts (Port 8107)
                           ├── AwesomeEWOLambo (Port 8108)
                           ├── AwesomeEWOLambo_Shorts (Port 8109)
                           ├── AlexNexusForgeV8AIV2 (Port 8112)
                           ├── GeneStrategy_v2 (Port 8114)
                           ├── GeneStrategy_v2_Shorts (Port 8115)
                           ├── KamaFama (Port 8091)
                           ├── KamaFama_Shorts (Port 8093)
                           ├── FrankenStrat (Port 8119)
                           ├── FrankenStrat_Shorts (Port 8118)
                           ├── BB_RPB_TSL (Port 8122)
                           ├── BB_RPB_TSL_Shorts (Port 8123)
                           ├── SimpleRSI (Port 8124)
                           └── SimpleRSI_Shorts (Port 8125)
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
- `falcontrader.env` - FalconTrader strategy (longs with 3x leverage, DCA, multi-entry)
- `falcontrader_short.env` - FalconTrader_Short strategy (shorts with 3x leverage, DCA, same settings as longs)
- `e0v1e.env` - E0V1E strategy (longs-only with 3x leverage)
- `e0v1e_shorts.env` - E0V1E_Shorts strategy (shorts-only with 3x leverage)
- `auto_ei_t4c0s.env` - Auto_EI_t4c0s strategy (longs, weighted EWO scoring)
- `auto_ei_t4c0s_shorts.env` - Auto_EI_t4c0s_Shorts strategy (shorts with 3x leverage)
- `etcg.env` - ETCG strategy (longs-only, multi-entry)
- `etcg_shorts.env` - ETCG_Shorts strategy (shorts-only, multi-entry)
- `cluchanix_hhll.env` - ClucHAnix_hhll strategy (longs-only, Heikin Ashi + BB)
- `cluchanix_hhll_shorts.env` - ClucHAnix_hhll_Shorts strategy (shorts-only, Heikin Ashi + BB)
- `awesomeewolambo.env` - AwesomeEWOLambo strategy (longs-only)
- `awesomeewolambo_shorts.env` - AwesomeEWOLambo_Shorts strategy (shorts-only)
- `alexnexusforgev8aiv2.env` - AlexNexusForgeV8AIV2 strategy (long + short capable)
- `genestrategy_v2.env` - GeneStrategy_v2 strategy (longs with 3x leverage + DCA)
- `genestrategy_v2_shorts.env` - GeneStrategy_v2_Shorts strategy (shorts with 3x leverage + DCA)
- `kamafama.env` - KamaFama strategy (longs with 3x leverage, KAMA/FAMA mean-reversion)
- `kamafama_shorts.env` - KamaFama_Shorts strategy (shorts with 3x leverage, KAMA/FAMA mean-reversion)
- `frankenstrat.env` - FrankenStrat strategy (longs with 3x leverage, multi-signal dip buyer)
- `frankenstrat_shorts.env` - FrankenStrat_Shorts strategy (shorts with 3x leverage, multi-signal inverted)
- `bb_rpb_tsl.env` - BB_RPB_TSL strategy (longs with 3x leverage, 19-signal BB/EWO dip buyer)
- `bb_rpb_tsl_shorts.env` - BB_RPB_TSL_Shorts strategy (shorts with 3x leverage, inverted 19-signal rally shorter)
- `simplersi.env` - SimpleRSI strategy (longs with 3x leverage, RSI momentum breakout on 1d)
- `simplersi_shorts.env` - SimpleRSI_Shorts strategy (shorts with 3x leverage, RSI momentum breakdown on 1d)

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
| **FalconTrader** | `Vasko_FalconTrader` | `http://freq.gaiaderma.com/falcontrader` | `falcontrader_user` | `falcontrader_secure_password` |
| **FalconTrader_Short** | `Vasko_FalconTrader_Shorts` | `http://freq.gaiaderma.com/falcontrader_short` | `falcontrader_short_user` | `falcontrader_short_secure_password` |
| **E0V1E** | `Vasko_E0V1E` | `http://freq.gaiaderma.com/e0v1e` | `e0v1e_user` | `e0v1e_secure_password` |
| **E0V1E_Shorts** | `Vasko_E0V1E_Shorts` | `http://freq.gaiaderma.com/e0v1e_shorts` | `e0v1e_shorts_user` | `e0v1e_shorts_secure_password` |
| **Auto_EI_t4c0s** | `Vasko_Auto_EI_t4c0s` | `http://freq.gaiaderma.com/auto_ei_t4c0s` | `auto_ei_t4c0s_user` | `auto_ei_t4c0s_secure_password` |
| **Auto_EI_t4c0s_Shorts** | `Vasko_Auto_EI_t4c0s_Shorts` | `http://freq.gaiaderma.com/auto_ei_t4c0s_shorts` | `auto_ei_t4c0s_shorts_user` | `auto_ei_t4c0s_shorts_secure_password` |
| **ETCG** | `Vasko_ETCG` | `http://freq.gaiaderma.com/etcg` | `etcg_user` | `etcg_secure_password` |
| **ETCG_Shorts** | `Vasko_ETCG_Shorts` | `http://freq.gaiaderma.com/etcg_shorts` | `etcg_shorts_user` | `etcg_shorts_secure_password` |
| **ClucHAnix_hhll** | `Vasko_ClucHAnix_hhll` | `http://freq.gaiaderma.com/cluchanix_hhll` | `cluchanix_hhll_user` | `cluchanix_hhll_secure_password` |
| **ClucHAnix_hhll_Shorts** | `Vasko_ClucHAnix_hhll_Shorts` | `http://freq.gaiaderma.com/cluchanix_hhll_shorts` | `cluchanix_hhll_shorts_user` | `cluchanix_hhll_shorts_secure_password` |
| **AwesomeEWOLambo** | `Vasko_AwesomeEWOLambo` | `http://freq.gaiaderma.com/awesomeewolambo` | `awesomeewolambo_user` | `awesomeewolambo_secure_password` |
| **AwesomeEWOLambo_Shorts** | `Vasko_AwesomeEWOLambo_Shorts` | `http://freq.gaiaderma.com/awesomeewolambo_shorts` | `awesomeewolambo_shorts_user` | `awesomeewolambo_shorts_secure_password` |
| **AlexNexusForgeV8AIV2** | `Vasko_AlexNexusForgeV8AIV2` | `http://freq.gaiaderma.com/alexnexusforgev8aiv2` | `alexnexusforgev8aiv2_user` | `alexnexusforgev8aiv2_secure_password` |
| **GeneStrategy_v2** | `Vasko_GeneStrategy_v2` | `http://freq.gaiaderma.com/genestrategy_v2` | `genestrategy_v2_user` | `genestrategy_v2_secure_password` |
| **GeneStrategy_v2_Shorts** | `Vasko_GeneStrategy_v2_Shorts` | `http://freq.gaiaderma.com/genestrategy_v2_shorts` | `genestrategy_v2_shorts_user` | `genestrategy_v2_shorts_secure_password` |
| **KamaFama** | `Vasko_KamaFama` | `http://freq.gaiaderma.com/kamafama` | `kamafama_user` | `kamafama_secure_password` |
| **KamaFama_Shorts** | `Vasko_KamaFama_Shorts` | `http://freq.gaiaderma.com/kamafama_shorts` | `kamafama_shorts_user` | `kamafama_shorts_secure_password` |
| **FrankenStrat** | `Vasko_FrankenStrat` | `http://freq.gaiaderma.com/frankenstrat` | `frankenstrat_user` | `frankenstrat_secure_password` |
| **FrankenStrat_Shorts** | `Vasko_FrankenStrat_Shorts` | `http://freq.gaiaderma.com/frankenstrat_shorts` | `frankenstrat_shorts_user` | `frankenstrat_shorts_secure_password` |
| **BB_RPB_TSL** | `Vasko_BB_RPB_TSL` | `http://freq.gaiaderma.com/bb_rpb_tsl` | `bb_rpb_tsl_user` | `bb_rpb_tsl_secure_password` |
| **BB_RPB_TSL_Shorts** | `Vasko_BB_RPB_TSL_Shorts` | `http://freq.gaiaderma.com/bb_rpb_tsl_shorts` | `bb_rpb_tsl_shorts_user` | `bb_rpb_tsl_shorts_secure_password` |
| **SimpleRSI** | `Vasko_SimpleRSI` | `http://freq.gaiaderma.com/simplersi` | `simplersi_user` | `simplersi_secure_password` |
| **SimpleRSI_Shorts** | `Vasko_SimpleRSI_Shorts` | `http://freq.gaiaderma.com/simplersi_shorts` | `simplersi_shorts_user` | `simplersi_shorts_secure_password` |

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
curl http://127.0.0.1:8092/api/v1/ping  # FalconTrader
curl http://127.0.0.1:8094/api/v1/ping  # FalconTrader_Short
curl http://127.0.0.1:8098/api/v1/ping  # E0V1E
curl http://127.0.0.1:8099/api/v1/ping  # E0V1E_Shorts
curl http://127.0.0.1:8100/api/v1/ping  # Auto_EI_t4c0s
curl http://127.0.0.1:8101/api/v1/ping  # Auto_EI_t4c0s_Shorts
curl http://127.0.0.1:8102/api/v1/ping  # ETCG
curl http://127.0.0.1:8103/api/v1/ping  # ETCG_Shorts
curl http://127.0.0.1:8106/api/v1/ping  # ClucHAnix_hhll
curl http://127.0.0.1:8107/api/v1/ping  # ClucHAnix_hhll_Shorts
curl http://127.0.0.1:8108/api/v1/ping  # AwesomeEWOLambo
curl http://127.0.0.1:8109/api/v1/ping  # AwesomeEWOLambo_Shorts
curl http://127.0.0.1:8112/api/v1/ping  # AlexNexusForgeV8AIV2
curl http://127.0.0.1:8114/api/v1/ping  # GeneStrategy_v2
curl http://127.0.0.1:8115/api/v1/ping  # GeneStrategy_v2_Shorts
curl http://127.0.0.1:8091/api/v1/ping  # KamaFama
curl http://127.0.0.1:8093/api/v1/ping  # KamaFama_Shorts
curl http://127.0.0.1:8119/api/v1/ping  # FrankenStrat
curl http://127.0.0.1:8118/api/v1/ping  # FrankenStrat_Shorts
curl http://127.0.0.1:8122/api/v1/ping  # BB_RPB_TSL
curl http://127.0.0.1:8123/api/v1/ping  # BB_RPB_TSL_Shorts
curl http://127.0.0.1:8124/api/v1/ping  # SimpleRSI
curl http://127.0.0.1:8125/api/v1/ping  # SimpleRSI_Shorts

# Test through NGINX
curl http://freq.gaiaderma.com/nfi-x7/api/v1/ping
curl http://freq.gaiaderma.com/falcontrader/api/v1/ping
curl http://freq.gaiaderma.com/falcontrader_short/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e_shorts/api/v1/ping
curl http://freq.gaiaderma.com/auto_ei_t4c0s/api/v1/ping
curl http://freq.gaiaderma.com/auto_ei_t4c0s_shorts/api/v1/ping
curl http://freq.gaiaderma.com/etcg/api/v1/ping
curl http://freq.gaiaderma.com/etcg_shorts/api/v1/ping
curl http://freq.gaiaderma.com/cluchanix_hhll/api/v1/ping
curl http://freq.gaiaderma.com/cluchanix_hhll_shorts/api/v1/ping
curl http://freq.gaiaderma.com/awesomeewolambo/api/v1/ping
curl http://freq.gaiaderma.com/awesomeewolambo_shorts/api/v1/ping
curl http://freq.gaiaderma.com/alexnexusforgev8aiv2/api/v1/ping
curl http://freq.gaiaderma.com/genestrategy_v2/api/v1/ping
curl http://freq.gaiaderma.com/genestrategy_v2_shorts/api/v1/ping
curl http://freq.gaiaderma.com/kamafama/api/v1/ping
curl http://freq.gaiaderma.com/kamafama_shorts/api/v1/ping
curl http://freq.gaiaderma.com/frankenstrat/api/v1/ping
curl http://freq.gaiaderma.com/frankenstrat_shorts/api/v1/ping
curl http://freq.gaiaderma.com/bb_rpb_tsl/api/v1/ping
curl http://freq.gaiaderma.com/bb_rpb_tsl_shorts/api/v1/ping
curl http://freq.gaiaderma.com/simplersi/api/v1/ping
curl http://freq.gaiaderma.com/simplersi_shorts/api/v1/ping
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
- `falcontrader.log`, `falcontrader_short.log`
- `e0v1e.log`, `e0v1e_shorts.log`
- `auto_ei_t4c0s.log`, `auto_ei_t4c0s_shorts.log`
- `etcg.log`, `etcg_shorts.log`
- `kamafama.log`, `kamafama_shorts.log`
- etc.

## Configuration Details

### Shared Configuration
All strategies use the same base configuration (`user_data/strategies/config.json`) but can be customized via environment variables.

### Port Allocation

| Port | Strategy | Direction | Leverage |
|------|----------|-----------|----------|
| 8080 | nfi-x7 | Longs | - |
| 8092 | FalconTrader | Longs | 3x |
| 8094 | FalconTrader_Short | Shorts | 3x |
| 8098 | E0V1E | Longs | 3x |
| 8099 | E0V1E_Shorts | Shorts | 3x |
| 8100 | Auto_EI_t4c0s | Longs | - |
| 8101 | Auto_EI_t4c0s_Shorts | Shorts | 3x |
| 8102 | ETCG | Longs | 3x |
| 8103 | ETCG_Shorts | Shorts | 3x |
| 8106 | ClucHAnix_hhll | Longs | - |
| 8107 | ClucHAnix_hhll_Shorts | Shorts | - |
| 8108 | AwesomeEWOLambo | Longs | - |
| 8109 | AwesomeEWOLambo_Shorts | Shorts | - |
| 8112 | AlexNexusForgeV8AIV2 | Long + Shorts | - |
| 8114 | GeneStrategy_v2 | Longs | 3x |
| 8115 | GeneStrategy_v2_Shorts | Shorts | 3x |
| 8091 | KamaFama | Longs | 3x |
| 8093 | KamaFama_Shorts | Shorts | 3x |
| 8118 | FrankenStrat_Shorts | Shorts | 3x |
| 8119 | FrankenStrat | Longs | 3x |
| 8122 | BB_RPB_TSL | Longs | 3x |
| 8123 | BB_RPB_TSL_Shorts | Shorts | 3x |
| 8124 | SimpleRSI | Longs | 3x |
| 8125 | SimpleRSI_Shorts | Shorts | 3x |

**Freed ports** (available for future strategies): 8097, 8104, 8126+

### Database Separation
Each strategy uses its own SQLite database:
- `nfi-x7-tradesv3.sqlite`
- `falcontrader-tradesv3.sqlite`
- `falcontrader_short-tradesv3.sqlite`
- `e0v1e-tradesv3.sqlite`
- `e0v1e_shorts-tradesv3.sqlite`
- `auto_ei_t4c0s-tradesv3.sqlite`
- `auto_ei_t4c0s_shorts-tradesv3.sqlite`
- `etcg-tradesv3.sqlite`
- `etcg_shorts-tradesv3.sqlite`
- `cluchanix_hhll-tradesv3.sqlite`
- `cluchanix_hhll_shorts-tradesv3.sqlite`
- `awesomeewolambo-tradesv3.sqlite`
- `awesomeewolambo_shorts-tradesv3.sqlite`
- `alexnexusforgev8aiv2-tradesv3.sqlite`
- `genestrategy_v2-tradesv3.sqlite`
- `genestrategy_v2_shorts-tradesv3.sqlite`
- `kamafama-tradesv3.sqlite`
- `kamafama_shorts-tradesv3.sqlite`
- `frankenstrat-tradesv3.sqlite`
- `frankenstrat_shorts-tradesv3.sqlite`
- `bb_rpb_tsl-tradesv3.sqlite`
- `bb_rpb_tsl_shorts-tradesv3.sqlite`
- `simplersi-tradesv3.sqlite`
- `simplersi_shorts-tradesv3.sqlite`

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

## Removed Strategies (February 2026)

The following strategies were removed due to poor performance in bear market conditions:

| Strategy | Port | Reason |
|----------|------|--------|
| ElliotV5_SMA | 8091 | Poor 2-month performance |
| ElliotV5_SMA_Shorts | 8097 | Short counterpart also underperforming |
| NASOSv4 | 8093 | Poor performance |
| NASOSv4_Shorts | 8104 | Removed with parent |
| RsiquiV5 | 8094 | Poor performance |
| ElliotV5HO | 8112 | Poor performance (port now reused by AlexNexusForgeV8AIV2) |
| ElliotV5HO_Shorts | 8113 | Short counterpart underperforming |
| EI4_t4c0s_V2_2 | 8100 | Replaced by Auto_EI_t4c0s |
| EI4_t4c0s_V2_2_Shorts | 8101 | Replaced by Auto_EI_t4c0s_Shorts |

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

**Last Updated**: February 11, 2026

## Recent Changes (February 11, 2026)
- **Added**: BB_RPB_TSL (longs, port 8122) - 19-signal BB/EWO dip buyer with custom trailing stoploss, 3x leverage
- **Added**: BB_RPB_TSL_Shorts (shorts, port 8123) - Shorts-only variant with inverted 19 entry signals, 3x leverage, max 4 shorts
- **Added**: SimpleRSI (longs, port 8124) - RSI momentum breakout on 1d timeframe, 3x leverage
- **Added**: SimpleRSI_Shorts (shorts, port 8125) - Shorts-only RSI momentum breakdown, 3x leverage, max 4 shorts

## Recent Changes (February 10, 2026)
- **Added**: FrankenStrat (longs, port 8119) - Multi-signal dip buyer with 3x leverage, SSL/MACD/EWO-based entries
- **Added**: FrankenStrat_Shorts (shorts, port 8118) - Shorts-only variant of FrankenStrat with 7 inverted signals, 3x leverage, -20% emergency backstop
- **Fixed**: `np.NAN` → `np.nan` in both FrankenStrat files (NumPy 2.x compatibility)

## Recent Changes (February 9, 2026)
- **Removed**: BinClucMadV1 strategy (port 8092) - replaced by FalconTrader
- **Added**: FalconTrader (longs, port 8092) - Multi-entry EWO/DCA strategy with 14+ entry signals and 3x leverage
- **Added**: FalconTrader_Short (shorts, port 8094) - Shorts variant with 3x leverage, same settings as longs, just inverted logic
