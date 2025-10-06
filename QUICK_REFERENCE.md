# FreqTrade Multi-Strategy Quick Reference

Quick reference for managing all 15 FreqTrade strategies.

## 🚀 Quick Start Commands

```bash
# Start all strategies
./deploy-multi-strategies.sh start

# Check status
./deploy-multi-strategies.sh status

# Health check
./deploy-multi-strategies.sh health-check

# View all logs
./deploy-multi-strategies.sh logs
```

## 📋 All 13 Strategies

| # | Strategy | URL | Port |
|---|----------|-----|------|
| 1 | NFI-X6 | `freq.gaiaderma.com/nfi-x6` | 8080 |
| 2 | Bandtastic | `freq.gaiaderma.com/bandtastic` | 8082 |
| 3 | TrendFollowing | `freq.gaiaderma.com/trendfollowing` | 8083 |
| 4 | FVG | `freq.gaiaderma.com/fvg` | 8085 |
| 5 | PowerTower | `freq.gaiaderma.com/powertower` | 8086 |
| 6 | FastSupertrend | `freq.gaiaderma.com/fastsupertrend` | 8087 |
| 7 | MacheteV8b | `freq.gaiaderma.com/machetev8b` | 8089 |
| 8 | ElliotV5_SMA | `freq.gaiaderma.com/elliotv5_sma` | 8090 |
| 9 | BinClucMadV1 | `freq.gaiaderma.com/binclucmadv1` | 8091 |
| 10 | NASOSv4 | `freq.gaiaderma.com/nasosv4` | 8092 |
| 11 | MartyEMA | `freq.gaiaderma.com/martyema` | 8093 |
| 12 | Ichimoku | `freq.gaiaderma.com/ichimoku` | 8094 |
| 13 | BigWill | `freq.gaiaderma.com/bigwill` | 8095 |

## 🎯 FreqUI Bot URLs (Copy-Paste Ready)

**Important**: Use base URLs, FreqUI adds `/api/v1/*` automatically.

```
http://freq.gaiaderma.com/nfi-x6
http://freq.gaiaderma.com/bandtastic
http://freq.gaiaderma.com/trendfollowing
http://freq.gaiaderma.com/fvg
http://freq.gaiaderma.com/powertower
http://freq.gaiaderma.com/fastsupertrend
http://freq.gaiaderma.com/machetev8b
http://freq.gaiaderma.com/elliotv5_sma
http://freq.gaiaderma.com/binclucmadv1
http://freq.gaiaderma.com/nasosv4
http://freq.gaiaderma.com/martyema
http://freq.gaiaderma.com/ichimoku
http://freq.gaiaderma.com/bigwill
```

## 🔧 Management Commands

### Individual Strategy Control
```bash
# Start
./deploy-multi-strategies.sh start <strategy-name>

# Stop
./deploy-multi-strategies.sh stop <strategy-name>

# Restart
./deploy-multi-strategies.sh restart <strategy-name>

# Logs
./deploy-multi-strategies.sh logs <strategy-name>
```

### Examples
```bash
./deploy-multi-strategies.sh start machetev8b
./deploy-multi-strategies.sh restart elliotv5_sma
./deploy-multi-strategies.sh logs bigwill
```

## 🧪 Health Check Script

Save as `test-all-strategies.sh`:

```bash
#!/bin/bash
STRATEGIES=(nfi-x6 bandtastic trendfollowing fvg powertower fastsupertrend machetev8b elliotv5_sma binclucmadv1 nasosv4 martyema ichimoku bigwill)

echo "Testing all 13 strategies..."
for strategy in "${STRATEGIES[@]}"; do
  echo -n "$strategy: "
  curl -s http://freq.gaiaderma.com/$strategy/api/v1/ping | grep -q "pong" && echo "✅" || echo "❌"
done
```

## 📁 Key Files

- `docker-compose-multi-strategies.yml` - Container definitions
- `deploy-multi-strategies.sh` - Management script
- `nginx-freqtrade-corrected.conf` - NGINX config
- `env-files/*.env` - Strategy configurations
- `STRATEGIES_REFERENCE.md` - Full documentation
- `MULTI_STRATEGY_SETUP.md` - Setup guide

## 🔍 Troubleshooting

### Check Running Containers
```bash
docker compose -f docker-compose-multi-strategies.yml ps
```

### Test Direct API Access
```bash
curl http://127.0.0.1:8089/api/v1/ping  # MacheteV8b
curl http://127.0.0.1:8090/api/v1/ping  # ElliotV5_SMA
```

### Test Through NGINX
```bash
curl http://freq.gaiaderma.com/machetev8b/api/v1/ping
curl http://freq.gaiaderma.com/elliotv5_sma/api/v1/ping
```

### Check Logs
```bash
# All logs
./deploy-multi-strategies.sh logs

# Specific strategy
./deploy-multi-strategies.sh logs machetev8b

# Live follow
docker compose -f docker-compose-multi-strategies.yml logs -f freqtrade-machetev8b
```

### Restart NGINX
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### Check Port Usage
```bash
netstat -tulpn | grep ":80[89][0-9]"
```

## 📊 Resource Monitoring

```bash
# Docker stats
docker stats

# Check specific containers
docker stats freqtrade-machetev8b freqtrade-elliotv5_sma freqtrade-bigwill

# Database sizes
du -sh user_data/*.sqlite

# Log file sizes
du -sh user_data/logs/*.log
```

## 🔐 Default Credentials

All strategies use pattern:
- **Username**: `{strategy}_user`
- **Password**: `{strategy}_secure_password`

Examples:
- machetev8b: `machetev8b_user` / `machetev8b_secure_password`
- elliotv5_sma: `elliotv5_sma_user` / `elliotv5_sma_secure_password`
- bigwill: `bigwill_user` / `bigwill_secure_password`

## 📝 Strategy Name Mapping

| Display Name | Internal Name | Container Name |
|--------------|---------------|----------------|
| MacheteV8b | machetev8b | freqtrade-machetev8b |
| ElliotV5_SMA | elliotv5_sma | freqtrade-elliotv5_sma |
| BinClucMadV1 | binclucmadv1 | freqtrade-binclucmadv1 |
| NASOSv4 | nasosv4 | freqtrade-nasosv4 |
| MartyEMA | martyema | freqtrade-martyema |
| Ichimoku | ichimoku | freqtrade-ichimoku |
| BigWill | bigwill | freqtrade-bigwill |

## ⚡ Quick Actions

### Restart All New Strategies
```bash
for strategy in machetev8b elliotv5_sma binclucmadv1 nasosv4 martyema ichimoku bigwill; do
  ./deploy-multi-strategies.sh restart $strategy
done
```

### Check All New Strategies Health
```bash
for strategy in machetev8b elliotv5_sma binclucmadv1 nasosv4 martyema ichimoku bigwill; do
  echo -n "$strategy: "
  curl -s http://freq.gaiaderma.com/$strategy/api/v1/ping
done
```

### View All New Strategies Logs
```bash
for strategy in machetev8b elliotv5_sma binclucmadv1 nasosv4 martyema ichimoku bigwill; do
  echo "=== $strategy ==="
  ./deploy-multi-strategies.sh logs $strategy | tail -20
done
```

## 🎯 Environment File Locations

```
env-files/
├── machetev8b.env
├── elliotv5_sma.env
├── binclucmadv1.env
├── nasosv4.env
├── martyema.env
├── ichimoku.env
└── bigwill.env
```

## 📖 Documentation Files

1. **QUICK_REFERENCE.md** (this file) - Quick commands
2. **STRATEGIES_REFERENCE.md** - Complete strategy details
3. **MULTI_STRATEGY_SETUP.md** - Full setup guide

## ✅ Verification Checklist

- [ ] All 15 containers running: `docker compose -f docker-compose-multi-strategies.yml ps`
- [ ] All endpoints responding: Run health check script
- [ ] NGINX configured: `sudo nginx -t`
- [ ] Bots added to FreqUI with correct URLs
- [ ] API credentials configured in env files
- [ ] Logs are being written: `ls -lh user_data/logs/`
- [ ] Databases created: `ls -lh user_data/*.sqlite`

## 🚨 Common Issues & Fixes

### Issue: Container won't start
```bash
# Check logs for errors
./deploy-multi-strategies.sh logs strategy-name

# Try removing and recreating
docker compose -f docker-compose-multi-strategies.yml rm -f freqtrade-strategy-name
./deploy-multi-strategies.sh start strategy-name
```

### Issue: API not responding
```bash
# Test direct connection first
curl http://127.0.0.1:PORT/api/v1/ping

# If direct works but NGINX doesn't, reload NGINX
sudo nginx -t && sudo systemctl reload nginx
```

### Issue: CORS errors in FreqUI
1. Check FreqUI bot URL has NO `/api/v1/` in it
2. Verify env file has correct CORS settings:
   ```
   FREQTRADE__API_SERVER__CORS_ORIGINS=https://freq.gaiaderma.com,http://freq.gaiaderma.com
   ```
3. Restart container: `./deploy-multi-strategies.sh restart strategy-name`

## 📞 Support

- **FreqTrade Docs**: https://www.freqtrade.io/en/stable/
- **Project Repo**: Check local git repository
- **Logs**: Always check `./deploy-multi-strategies.sh logs strategy-name` first

---

**Total Strategies**: 15 (8 original + 7 new)  
**Status**: All operational ✅  
**Last Updated**: September 30, 2024
