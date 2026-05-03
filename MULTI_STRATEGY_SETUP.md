# Multi-Strategy FreqTrade Setup Guide

This guide will help you set up multiple FreqTrade strategies with NGINX reverse proxy, allowing you to manage all strategies through a single FreqUI interface.

## Overview

The multi-strategy setup includes:

- **29 active trading strategies** running in separate Docker containers
- **NGINX reverse proxy** for unified access with proper path routing
- **Individual environment configurations** for each strategy
- **Single FreqUI interface** to manage all bots
- **Health monitoring** and deployment scripts
- **Proper CORS configuration** for cross-origin requests

## Architecture

```
Internet → NGINX (Port 80) → FreqTrade Strategies
                           ├── nfi-x7 (Port 8080)
                           ├── FastSupertrend_optim3_rsi_70 (Port 8098)
                           ├── FastSupertrend_optim_quick3 (Port 8099)
                           ├── e0v1e_binance (Port 8114)
                           ├── e0v1e_binance_shorts (Port 8115)
                           ├── BinHV27_combined (Port 8092)
                           ├── Auto_EI_t4c0s (Port 8100)
                           ├── FibonacciEMATrendStrategy (Port 8103)
                           ├── GnF_V2 (Port 8091)
                           ├── ZaratustraV31 (Port 8118)
                           ├── ZaratustraDCA2_06 (Port 8119)
                           ├── BollingerBounce (Port 8124)
                           ├── BollingerBounce_Shorts (Port 8125)
                           ├── KeltnerBounce (Port 8126)
                           ├── KeltnerBounce_Shorts (Port 8127)
                           ├── UltraSmartStrategy_NoStoploss_v2 (Port 8128)
                           ├── FenixTopProfit (Port 8129)
                           ├── MtfScalper (Port 8131)
                           ├── AlexBattleTankKillerV4 (Port 8132)
                           ├── TripleSuperTrendADXRSI (Port 8134)
                           ├── Best5m (Port 8135)
                           ├── DevDsl2Approx (Port 8136)
                           ├── Picasso CE/CTI/STC/EMA (Port 8137)
                           ├── CombinedBinHAndClucV8 (Port 8139)
                           ├── CombinedBinHAndClucV8XH (Port 8140)
                           ├── newstrategy4_dca (Port 8141)
                           ├── OsirisXRSI (Port 8142)
                           └── AdvancedFuturesSwingStrategy (Port 8143)
```

## Files

### Docker Configuration

- `docker-compose-multi-strategies.yml` - Multi-strategy Docker Compose file
- `deploy-multi-strategies.sh` - Deployment and management script
- `docker/Dockerfile.custom` - Shared custom Freqtrade image used by the full multi-strategy stack

### NGINX Configuration

- `nginx-freqtrade-multi.conf` - NGINX configuration with proper path routing
- `freqtrade-proxy-headers.conf` - Reusable proxy headers

### Environment Files (in `env-files/`)

- `nfi-x7.env` - NostalgiaForInfinityX7 strategy
- `e0v1e.env` - FastSupertrend_optim3_rsi_70 runtime env file
- `e0v1e_shorts.env` - FastSupertrend_optim_quick3 runtime env file
- `e0v1e_binance.env` - E0V1E Binance-tuned isolated variant
- `e0v1e_binance_shorts.env` - E0V1E Binance-tuned shorts isolated dry-run variant
- `binhv27.env` - BinHV27 combined strategy
- `auto_ei_t4c0s.env` - Auto_EI_t4c0s strategy (longs, weighted EWO scoring)
- `fibonacciematrend.env` - FibonacciEMATrendStrategy (1h/4h EMA trend strategy, longs + shorts, no FreqAI)
- `gnf_v2.env` - GnF_V2 strategy (1h longs + shorts)
- `zaratustrav31.env` - ZaratustraV31 strategy (1h longs + shorts)
- `zaratustra.env` - ZaratustraDCA2_06 strategy (longs + shorts with DCA and protection logic)
- `bollingerbounce.env` - BollingerBounce strategy (longs with 3x leverage)
- `bollingerbounce_shorts.env` - BollingerBounce_Shorts strategy (shorts-only with 3x leverage)
- `keltnerbounce.env` - KeltnerBounce strategy (longs with 3x leverage)
- `keltnerbounce_shorts.env` - KeltnerBounce_Shorts strategy (shorts-only with 3x leverage)
- `ultrasmart_nostop_v2.env` - UltraSmartStrategy_NoStoploss_v2 strategy (long-only Lmao family strategy)
- `fenix.env` - FenixTopProfit strategy (longs + shorts 1h trend-following strategy)
- `mtfscalper.env` - MtfScalper strategy (multi-timeframe futures scalper)
- `alexbattletankkiller_v4.env` - AlexBattleTankKillerV4 strategy (patched dry-run validation rollout)
- `triplesupertrendadxrsi.env` - TripleSuperTrendADXRSI strategy (longs + shorts, triple Supertrend with ADX/RSI confirmation)
- `best5m.env` - Best5m strategy (5m SMA/RSI futures strategy, longs + shorts)
- `devdsl2approx.env` - DevDsl2Approx strategy (5m Bybit futures dry-run strategy, longs + shorts)
- `edtma.env` - EDTMA strategy (longs + shorts dry-run evaluation on Bybit futures, max_open_trades=3)
- `combinedbinhandclucv8.env` - CombinedBinHAndClucV8 strategy (5m Cluc/BinHV hybrid futures strategy)
- `combinedbinhandclucv8xh.env` - CombinedBinHAndClucV8XH strategy (5m Cluc/BinHV hybrid futures strategy, XH variant)
- `newstrategy4_dca.env` - newstrategy4 dry-run evaluation strategy (5m long-only Bybit futures strategy)
- `osirisxrsi.env` - OsirisXRSI (5m Bybit futures dry-run strategy, max_open_trades=2)
- `advancedfuturesswingstrategy.env` - AdvancedFuturesSwingStrategy (15m Bybit futures swing strategy, longs + shorts)

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

| Strategy                             | Bot Name                       | API URL                                            | Username                      | Password                                 |
| ------------------------------------ | ------------------------------ | -------------------------------------------------- | ----------------------------- | ---------------------------------------- |
| **nfi-x7**                           | `Vasko_NFI_X7`                 | `http://freq.gaiaderma.com/nfi-x7`                 | `nfi_x6_user`                 | `nfi_x6_secure_password`                 |
| **FastSupertrend_optim3_rsi_70**     | `Vasko_FastSupertrend_rsi_70`  | `http://freq.gaiaderma.com/fastsupertrend_rsi_70`  | `e0v1e_user`                  | `e0v1e_secure_password`                  |
| **FastSupertrend_optim_quick3**      | `Vasko_FastSupertrend_quick3`  | `http://freq.gaiaderma.com/fastsupertrend_quick3`  | `e0v1e_shorts_user`           | `e0v1e_shorts_secure_password`           |
| **e0v1e_binance**                    | `Vasko_E0V1E_Binance`          | `http://freq.gaiaderma.com/e0v1e_binance`          | `e0v1e_binance_user`          | `e0v1e_binance_secure_password`          |
| **e0v1e_binance_shorts**             | `Vasko_E0V1E_Binance_Shorts`   | `http://freq.gaiaderma.com/e0v1e_binance_shorts`   | `e0v1e_binance_shorts_user`   | `e0v1e_binance_shorts_secure_password`   |
| **BinHV27_combined**                 | `Vasko_BinHV27`                | `http://freq.gaiaderma.com/binhv27`                | `binhv27_user`                | `binhv27_secure_password`                |
| **Auto_EI_t4c0s**                    | `Vasko_Auto_EI_t4c0s`          | `http://freq.gaiaderma.com/auto_ei_t4c0s`          | `auto_ei_t4c0s_user`          | `auto_ei_t4c0s_secure_password`          |
| **FibonacciEMATrendStrategy**        | `Vasko_FibonacciEMATrend`      | `http://freq.gaiaderma.com/fibonacciematrend`      | `fibonacciematrend_user`      | `fibonacciematrend_secure_password`      |
| **GnF_V2**                           | `Vasko_GnF_V2`                 | `http://freq.gaiaderma.com/gnf_v2`                 | `gnf_v2_user`                 | `gnf_v2_secure_password`                 |
| **ZaratustraV31**                    | `Vasko_ZaratustraV31`          | `http://freq.gaiaderma.com/zaratustrav31`          | `zaratustrav31_user`          | `zaratustrav31_secure_password`          |
| **ZaratustraDCA2_06**                | `Vasko_ZaratustraDCA2_06`      | `http://freq.gaiaderma.com/zaratustra`             | `zaratustra_user`             | `zaratustra_secure_password`             |
| **BollingerBounce**                  | `Vasko_BollingerBounce`        | `http://freq.gaiaderma.com/bollingerbounce`        | `bollingerbounce_user`        | `bollingerbounce_secure_password`        |
| **BollingerBounce_Shorts**           | `Vasko_BollingerBounce_Shorts` | `http://freq.gaiaderma.com/bollingerbounce_shorts` | `bollingerbounce_shorts_user` | `bollingerbounce_shorts_secure_password` |
| **KeltnerBounce**                    | `Vasko_KeltnerBounce`          | `http://freq.gaiaderma.com/keltnerbounce`          | `keltnerbounce_user`          | `keltnerbounce_secure_password`          |
| **KeltnerBounce_Shorts**             | `Vasko_KeltnerBounce_Shorts`   | `http://freq.gaiaderma.com/keltnerbounce_shorts`   | `keltnerbounce_shorts_user`   | `keltnerbounce_shorts_secure_password`   |
| **UltraSmartStrategy_NoStoploss_v2** | `Vasko_UltraSmart_NoStop_v2`   | `http://freq.gaiaderma.com/ultrasmart_nostop_v2`   | `ultrasmart_nostop_v2_user`   | `ultrasmart_nostop_v2_secure_password`   |
| **FenixTopProfit**                   | `Vasko_FenixTopProfit`         | `http://freq.gaiaderma.com/fenix`                  | `fenix_user`                  | `fenix_secure_password`                  |
| **MtfScalper**                       | `Vasko_MtfScalper`             | `http://freq.gaiaderma.com/mtfscalper`             | `mtfscalper_user`             | `mtfscalper_secure_password`             |
| **AlexBattleTankKillerV4**           | `Vasko_AlexBattleTankKillerV4` | `http://freq.gaiaderma.com/alexbattletankkiller_v4` | `alexbattletankkiller_v4_user` | `alexbattletankkiller_v4_secure_password` |
| **TripleSuperTrendADXRSI**           | `Vasko_TripleSuperTrendADXRSI` | `http://freq.gaiaderma.com/triplesupertrendadxrsi` | `triplesupertrendadxrsi_user` | `triplesupertrendadxrsi_secure_password` |
| **Best5m**                           | `Vasko_Best5m`                 | `http://freq.gaiaderma.com/best5m`                 | `best5m_user`                 | `best5m_secure_password`                 |
| **DevDsl2Approx**                    | `Vasko_DevDsl2Approx`          | `http://freq.gaiaderma.com/devdsl2approx`          | `devdsl2approx_user`          | `devdsl2approx_secure_password`          |
| **EDTMA**                            | `Vasko_EDTMA`                  | `http://freq.gaiaderma.com/edtma`                  | `edtma_user`                  | `edtma_secure_password`                  |
| **CombinedBinHAndClucV8**            | `Vasko_CombinedBinHAndClucV8`  | `http://freq.gaiaderma.com/combinedbinhandclucv8`  | `combinedbinhandclucv8_user`  | `combinedbinhandclucv8_secure_password`  |
| **CombinedBinHAndClucV8XH**          | `Vasko_CombinedBinHAndClucV8XH` | `http://freq.gaiaderma.com/combinedbinhandclucv8xh` | `combinedbinhandclucv8xh_user` | `combinedbinhandclucv8xh_secure_password` |
| **newstrategy4**                     | `Vasko_newstrategy4_dca`       | `http://freq.gaiaderma.com/newstrategy4_dca`       | `newstrategy4_dca_user`       | `newstrategy4_dca_secure_password`       |
| **OsirisXRSI**                       | `Vasko_OsirisXRSI`             | `http://freq.gaiaderma.com/osirisxrsi`             | `osirisxrsi_user`             | `osirisxrsi_secure_password`             |
| **AdvancedFuturesSwingStrategy**     | `Vasko_AdvancedFuturesSwingStrategy` | `http://freq.gaiaderma.com/advancedfuturesswingstrategy` | `advancedfuturesswingstrategy_user` | `advancedfuturesswingstrategy_secure_password` |

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
curl http://127.0.0.1:8098/api/v1/ping  # FastSupertrend_optim3_rsi_70
curl http://127.0.0.1:8099/api/v1/ping  # FastSupertrend_optim_quick3
curl http://127.0.0.1:8114/api/v1/ping  # e0v1e_binance
curl http://127.0.0.1:8115/api/v1/ping  # e0v1e_binance_shorts
curl http://127.0.0.1:8092/api/v1/ping  # BinHV27_combined
curl http://127.0.0.1:8100/api/v1/ping  # Auto_EI_t4c0s
curl http://127.0.0.1:8103/api/v1/ping  # FibonacciEMATrendStrategy
curl http://127.0.0.1:8091/api/v1/ping  # GnF_V2
curl http://127.0.0.1:8118/api/v1/ping  # ZaratustraV31
curl http://127.0.0.1:8119/api/v1/ping  # ZaratustraDCA2_06
curl http://127.0.0.1:8124/api/v1/ping  # BollingerBounce
curl http://127.0.0.1:8125/api/v1/ping  # BollingerBounce_Shorts
curl http://127.0.0.1:8126/api/v1/ping  # KeltnerBounce
curl http://127.0.0.1:8127/api/v1/ping  # KeltnerBounce_Shorts
curl http://127.0.0.1:8128/api/v1/ping  # UltraSmartStrategy_NoStoploss_v2
curl http://127.0.0.1:8129/api/v1/ping  # FenixTopProfit
curl http://127.0.0.1:8131/api/v1/ping  # MtfScalper
curl http://127.0.0.1:8132/api/v1/ping  # AlexBattleTankKillerV4
curl http://127.0.0.1:8134/api/v1/ping  # TripleSuperTrendADXRSI
curl http://127.0.0.1:8135/api/v1/ping  # Best5m
curl http://127.0.0.1:8136/api/v1/ping  # DevDsl2Approx
curl http://127.0.0.1:8137/api/v1/ping  # Picasso CE/CTI/STC/EMA
curl http://127.0.0.1:8139/api/v1/ping  # CombinedBinHAndClucV8
curl http://127.0.0.1:8140/api/v1/ping  # CombinedBinHAndClucV8XH
curl http://127.0.0.1:8141/api/v1/ping  # newstrategy4_dca
curl http://127.0.0.1:8142/api/v1/ping  # OsirisXRSI
curl http://127.0.0.1:8143/api/v1/ping  # AdvancedFuturesSwingStrategy
# Test through NGINX
curl http://freq.gaiaderma.com/nfi-x7/api/v1/ping
curl http://freq.gaiaderma.com/fastsupertrend_rsi_70/api/v1/ping
curl http://freq.gaiaderma.com/fastsupertrend_quick3/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e_binance/api/v1/ping
curl http://freq.gaiaderma.com/e0v1e_binance_shorts/api/v1/ping
curl http://freq.gaiaderma.com/binhv27/api/v1/ping
curl http://freq.gaiaderma.com/auto_ei_t4c0s/api/v1/ping
curl http://freq.gaiaderma.com/fibonacciematrend/api/v1/ping
curl http://freq.gaiaderma.com/gnf_v2/api/v1/ping
curl http://freq.gaiaderma.com/zaratustrav31/api/v1/ping
curl http://freq.gaiaderma.com/zaratustra/api/v1/ping
curl http://freq.gaiaderma.com/bollingerbounce/api/v1/ping
curl http://freq.gaiaderma.com/bollingerbounce_shorts/api/v1/ping
curl http://freq.gaiaderma.com/keltnerbounce/api/v1/ping
curl http://freq.gaiaderma.com/keltnerbounce_shorts/api/v1/ping
curl http://freq.gaiaderma.com/ultrasmart_nostop_v2/api/v1/ping
curl http://freq.gaiaderma.com/fenix/api/v1/ping
curl http://freq.gaiaderma.com/mtfscalper/api/v1/ping
curl http://freq.gaiaderma.com/alexbattletankkiller_v4/api/v1/ping
curl http://freq.gaiaderma.com/triplesupertrendadxrsi/api/v1/ping
curl http://freq.gaiaderma.com/best5m/api/v1/ping
curl http://freq.gaiaderma.com/devdsl2approx/api/v1/ping
curl http://freq.gaiaderma.com/edtma/api/v1/ping
curl http://freq.gaiaderma.com/combinedbinhandclucv8/api/v1/ping
curl http://freq.gaiaderma.com/combinedbinhandclucv8xh/api/v1/ping
curl http://freq.gaiaderma.com/newstrategy4_dca/api/v1/ping
curl http://freq.gaiaderma.com/advancedfuturesswingstrategy/api/v1/ping
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
- `fastsupertrend_rsi_70.log`, `fastsupertrend_quick3.log`
- `auto_ei_t4c0s.log`
- `fibonacciematrend.log`
- `gnf_v2.log`
- `zaratustrav31.log`
- `devdsl2approx.log`
- `combinedbinhandclucv8.log`
- `combinedbinhandclucv8xh.log`
- `newstrategy4_dca.log`
- `advancedfuturesswingstrategy.log`
- etc.

## Configuration Details

### Shared Configuration

All strategies use the same base configuration (`user_data/strategies/config.json`) but can be customized via environment variables.

### Port Allocation

| Port | Strategy                         | Direction      | Leverage         |
| ---- | -------------------------------- | -------------- | ---------------- |
| 8080 | nfi-x7                           | Longs          | -                |
| 8098 | FastSupertrend_optim3_rsi_70     | Longs + Shorts | Strategy-defined |
| 8099 | FastSupertrend_optim_quick3      | Longs + Shorts | Strategy-defined |
| 8114 | e0v1e_binance                    | Longs          | Strategy-defined |
| 8115 | e0v1e_binance_shorts             | Shorts         | Strategy-defined |
| 8092 | BinHV27_combined                 | Longs + Shorts | Strategy-defined |
| 8100 | Auto_EI_t4c0s                    | Longs          | -                |
| 8103 | FibonacciEMATrendStrategy        | Longs + Shorts | Strategy-defined |
| 8091 | GnF_V2                           | Longs + Shorts | Strategy-defined |
| 8118 | ZaratustraV31                    | Longs + Shorts | Strategy-defined |
| 8119 | ZaratustraDCA2_06                | Longs + Shorts | Config-defined   |
| 8124 | BollingerBounce                  | Longs          | 3x               |
| 8125 | BollingerBounce_Shorts           | Shorts         | 3x               |
| 8126 | KeltnerBounce                    | Longs          | 3x               |
| 8127 | KeltnerBounce_Shorts             | Shorts         | 3x               |
| 8128 | UltraSmartStrategy_NoStoploss_v2 | Longs          | Config-defined   |
| 8129 | FenixTopProfit                   | Longs + Shorts | Config-defined   |
| 8131 | MtfScalper                       | Longs + Shorts | Strategy-defined |
| 8132 | AlexBattleTankKillerV4           | Longs + Shorts | Strategy-defined |
| 8134 | TripleSuperTrendADXRSI           | Longs + Shorts | Strategy-defined |
| 8135 | Best5m                           | Longs + Shorts | Strategy-defined |
| 8136 | DevDsl2Approx                    | Longs + Shorts | Config-defined   |
| 8137 | Picasso CE/CTI/STC/EMA           | Longs + Shorts | Strategy-defined |
| 8139 | CombinedBinHAndClucV8            | Longs + Shorts | Strategy-defined |
| 8140 | CombinedBinHAndClucV8XH          | Longs + Shorts | Strategy-defined |
| 8141 | newstrategy4_dca                | Longs          | Config-defined   |
| 8142 | OsirisXRSI                       | Longs + Shorts | Config-defined   |
| 8143 | AdvancedFuturesSwingStrategy     | Longs + Shorts | Strategy-defined |

**Freed ports** (available for future strategies): 8097, 8104, 8112, 8130, 8133, 8144+

### Database Separation

Each strategy uses its own SQLite database:

- `nfi-x7-tradesv3.sqlite`
- `fastsupertrend_rsi_70-tradesv3.sqlite`
- `fastsupertrend_quick3-tradesv3.sqlite`
- `e0v1e_binance-tradesv3.sqlite`
- `e0v1e_binance_shorts-tradesv3.sqlite`
- `binhv27-tradesv3.sqlite`
- `auto_ei_t4c0s-tradesv3.sqlite`
- `fibonacciematrend-tradesv3.sqlite`
- `gnf_v2-tradesv3.sqlite`
- `zaratustrav31-tradesv3.sqlite`
- `zaratustra-tradesv3.sqlite`
- `bollingerbounce-tradesv3.sqlite`
- `bollingerbounce_shorts-tradesv3.sqlite`
- `keltnerbounce-tradesv3.sqlite`
- `keltnerbounce_shorts-tradesv3.sqlite`
- `ultrasmart_nostop_v2-tradesv3.sqlite`
- `fenix-tradesv3.sqlite`
- `mtfscalper-tradesv3.sqlite`
- `alexbattletankkiller_v4-tradesv3.sqlite`
- `best5m-tradesv3.sqlite`
- `devdsl2approx-tradesv3.sqlite`
- `combinedbinhandclucv8-tradesv3.sqlite`
- `combinedbinhandclucv8xh-tradesv3.sqlite`
- `newstrategy4_dca-tradesv3.sqlite`
- `advancedfuturesswingstrategy-tradesv3.sqlite`

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

**Last Updated**: May 3, 2026

## Recent Changes (May 3, 2026)

- **Added**: `AdvancedFuturesSwingStrategy` multistrategy dry-run slot (port `8143`, path `/advancedfuturesswingstrategy`)
- **Added**: dedicated runtime env file, reverse-proxy route, health endpoint, and deploy-helper wiring for the new slot

## Recent Changes (April 19, 2026)

- **Added**: `newstrategy4_dca` multistrategy dry-run slot (port `8141`, path `/newstrategy4_dca`)
- **Validated**: `newstrategy4` top-5 Bybit futures backtest completed successfully after Freqtrade compatibility fixes in [`user_data/strategies/Zaratustra/newstrategy4_dca.py`](user_data/strategies/Zaratustra/newstrategy4_dca.py)

## Recent Changes (April 28, 2026)

- **Replaced**: AlexBandSniperV10AI with a patched AlexBattleTankKillerV4 in the port 8132 dry-run slot
- **Updated**: Port 8132 reverse-proxy path to `/alexbattletankkiller_v4`

## Recent Changes (May 1, 2026)

- **Added**: `GnF_V2` multistrategy dry-run slot (port `8091`, path `/gnf_v2`)
- **Added**: `ZaratustraV31` multistrategy dry-run slot (port `8118`, path `/zaratustrav31`)

## Recent Changes (March 18, 2026)

- **Added**: `e0v1e_binance` reverse-proxy and documentation sync (port 8114, path `/e0v1e_binance`)
- **Added**: `BinHV27_combined` multistrategy dry-run slot (port 8092, path `/binhv27`)

## Recent Changes (March 16, 2026)

- **Replaced**: AlexBandSniperV58COptuna with AlexBandSniperV10AI in the port 8132 dry-run slot
- **Removed**: SimpleRSI (port 8124) and SimpleRSI_Shorts (port 8125)
- **Added**: BollingerBounce (longs, port 8124) and BollingerBounce_Shorts (shorts, port 8125), both 3x leverage
- **Added**: KeltnerBounce (longs, port 8126) and KeltnerBounce_Shorts (shorts, port 8127), both 3x leverage

## Recent Changes (March 11, 2026)

- **Added**: ZaratustraDCA2_06 (mixed long/short, port 8119) with `/zaratustra` reverse-proxy routing

## Recent Changes (February 9, 2026)

- **Removed**: BinClucMadV1 strategy (port 8092)
