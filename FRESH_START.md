# Fresh Start Guide - Reset Production Trading Data

## Overview

This guide helps you completely reset your production trading data after liquidations or setup issues, while keeping all your configurations intact.

## What Gets Deleted

- ❌ **All trade history** (SQLite database files: `*-tradesv3.sqlite`)
- ❌ **All logs** (everything in `user_data/logs/`)
- ❌ **Open positions** (force-closed via API)
- ❌ **Lock files and temporary files**

## What Gets Kept

- ✅ **Strategy files** (all your custom strategies)
- ✅ **Configuration files** (env-files/, docker-compose.yml)
- ✅ **Docker setup** (images, containers recreated fresh)
- ✅ **Pairlist configurations**
- ✅ **API keys and credentials**

## Prerequisites

### 1. Push Leverage Fix to Production

The RsiquiV5 leverage has been changed from x10 → x3. Deploy it:

```bash
# On local Mac
cd /Users/vasmihay/Personal/AI_Projects/NostalgiaForInfinity
git add user_data/strategies/Rsiqui/
git commit -m "Fix RsiquiV5 leverage: x10 -> x3"
git push nfi-custom-strategies main

# On production server
ssh ubuntu@YOUR_SERVER
cd /opt/trading/nfi-custom-strategies
git pull origin main
```

### 2. Verify Blacklists Are Set

Make sure problematic pairs that caused liquidations are blacklisted in your env files:

```bash
# Check each env file
grep FREQTRADE__EXCHANGE__PAIR_BLACKLIST env-files/*.env
```

Should include pairs like those that caused issues.

---

## Step-by-Step Reset Process

### Step 1: Copy Reset Script to Production

```bash
# On local Mac
cd /Users/vasmihay/Personal/AI_Projects/NostalgiaForInfinity
scp reset-trading-data.sh ubuntu@YOUR_SERVER_IP:/opt/trading/nfi-custom-strategies/

# On production server
ssh ubuntu@YOUR_SERVER_IP
cd /opt/trading/nfi-custom-strategies
chmod +x reset-trading-data.sh
```

### Step 2: Review What Will Be Deleted

```bash
# See what databases exist
ls -lh user_data/*.sqlite

# See what logs exist
du -sh user_data/logs/

# Check if any bots have open positions
docker compose -f docker-compose-multi-strategies.yml exec freqtrade-nfi-x7 freqtrade show-trades --trade-ids --db-url sqlite:////freqtrade/user_data/nfi-x7-tradesv3.sqlite 2>/dev/null || echo "No trades"
```

### Step 3: Run the Reset Script

```bash
cd /opt/trading/nfi-custom-strategies
./reset-trading-data.sh
```

**The script will**:
1. Ask for confirmation (you must type `YES`)
2. Try to close all open positions
3. Stop all containers
4. Backup current databases and logs to `backups/reset-YYYYMMDD-HHMMSS/`
5. Delete all trading data
6. Start all containers fresh

**Expected output**:
```
========================================
🧹 Production Trading Data Reset
========================================

⚠️  WARNING: This will delete ALL trading data!
...

Are you sure you want to continue? Type 'YES' to confirm: YES

========================================
Step 1: Closing All Open Positions
========================================
[INFO] Found running containers, attempting to close positions...
...

========================================
✅ Reset Complete!
========================================
[SUCCESS] Trading data has been wiped and all bots restarted

Summary:
  ✅ All trade databases deleted
  ✅ All logs cleared
  ✅ Containers restarted with fresh state
  ✅ Backup saved to: backups/reset-20260125-143022
```

### Step 4: Verify Fresh Start

```bash
# Check no databases exist
ls user_data/*.sqlite
# Should show: No such file or directory

# Check containers are running
docker compose -f docker-compose-multi-strategies.yml ps
# All should show "running"

# Check logs for clean start
docker logs freqtrade-nfi-x7 | tail -20
# Should show fresh initialization

# Check leverage is correct for RsiquiV5
docker logs freqtrade-rsiquiv5 | grep -i leverage
# Should show leverage=3 (not 10)
```

### Step 5: Monitor First Trades

```bash
# Watch all logs
docker compose -f docker-compose-multi-strategies.yml logs -f

# Or watch specific strategy
docker logs -f freqtrade-nfi-x7

# Check for errors
docker compose -f docker-compose-multi-strategies.yml logs | grep -i error

# Monitor via FreqUI
# Open: http://freq.gaiaderma.com/nfi-x7 (and other strategies)
```

---

## Important Considerations

### First Trades After Reset

⚠️ **Bots have NO trade history**, so:
- DCA (Dollar Cost Averaging) strategies start fresh
- No position memory from before
- Risk management calculations start from zero
- May behave differently than mature bots

### Leverage Settings

After reset, verify each bot uses correct leverage:

| Strategy | Leverage | Notes |
|----------|----------|-------|
| NFI-X7 | x3-x5 | Check env file |
| ElliotV5_SMA | x3 | Check strategy file |
| **RsiquiV5** | **x3** | ✅ Fixed (was x10) |
| NASOSv4 | x3 | Check strategy file |
| E0V1E | x3 | Check strategy file |

### Blacklists

Ensure pairs that caused liquidations are blacklisted:

```bash
# Example blacklist format in env file
FREQTRADE__EXCHANGE__PAIR_BLACKLIST=["BTC/USDT:USDT","PROBLEMATIC/PAIR:USDT"]
```

---

## Troubleshooting

### If script fails at position closing:

Positions might still be open. Close manually:
```bash
# For each running bot, find its port and close positions
curl -X POST "http://127.0.0.1:8080/api/v1/forceexit/all" -H "Content-Type: application/json" -d '{}'

# Then re-run the script
./reset-trading-data.sh
```

### If containers won't start:

```bash
# Check logs
docker compose -f docker-compose-multi-strategies.yml logs

# Force recreate
docker compose -f docker-compose-multi-strategies.yml down
docker compose -f docker-compose-multi-strategies.yml up -d --force-recreate
```

### If you want to restore backup:

**NOT RECOMMENDED** (defeats purpose of fresh start), but if needed:

```bash
# Stop containers
docker compose -f docker-compose-multi-strategies.yml down

# Restore databases
cp backups/reset-YYYYMMDD-HHMMSS/databases/*.sqlite user_data/

# Restart
docker compose -f docker-compose-multi-strategies.yml up -d
```

### If leverage still wrong:

```bash
# Pull latest code
git pull origin main

# Verify leverage in code
grep -A2 "def leverage" user_data/strategies/Rsiqui/RsiquiV5.py
# Should show: return 3

# Force recreate containers to reload code
docker compose -f docker-compose-multi-strategies.yml up -d --force-recreate freqtrade-rsiquiv5
```

---

## After Reset Checklist

- [ ] All containers running: `docker compose -f docker-compose-multi-strategies.yml ps`
- [ ] No databases exist: `ls user_data/*.sqlite` (should be empty)
- [ ] Logs are fresh: `ls -la user_data/logs/` (new files only)
- [ ] RsiquiV5 leverage is x3: `grep "return" user_data/strategies/Rsiqui/RsiquiV5.py | grep leverage`
- [ ] Blacklists are set: `grep PAIR_BLACKLIST env-files/*.env`
- [ ] FreqUI accessible: http://freq.gaiaderma.com/nfi-x7
- [ ] No errors in logs: `docker compose -f docker-compose-multi-strategies.yml logs | grep -i error`
- [ ] First trades monitored closely

---

## Files Modified

**Leverage fixes** (local Mac, then pushed to production):
- `user_data/strategies/Rsiqui/RsiquiV5.py` - x10 → x3
- `user_data/strategies/Rsiqui/RsiquiV5_long_only.py` - x10 → x3
- `user_data/strategies/Rsiqui/RsiquiV2.py` - x10 → x3

**Reset script** (copy to production):
- `reset-trading-data.sh` - Safe data wipe script

---

## Summary

**Problem**: Liquidations during setup, bad trades, wrong leverage
**Solution**: Complete fresh start while keeping all configs

**What's deleted**: All trade history, logs, positions
**What's kept**: Strategies, configs, env files, Docker setup

**Result**: Clean slate, correct leverage, ready to trade safely

🎯 **Start fresh, trade smart!**
