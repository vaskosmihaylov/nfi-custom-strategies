# NFI Strategy Deployment - Simplified Guide

## Architecture

### Simple 3-Step Flow
```
1. Upstream (iterativv/NostalgiaForInfinity)
        ↓ (GitHub Actions syncs daily at 6 AM UTC)
2. Your Fork (vaskosmihaylov/nfi-custom-strategies)
        ↓ (Production pulls daily or on-demand)
3. Production Server (Amazon Ubuntu)
```

### Directory Structure

**Root level** (from upstream):
```
NostalgiaForInfinityX7.py
NostalgiaForInfinityX6.py
NostalgiaForInfinityX5.py
```

**user_data/strategies/** (mixed):
```
user_data/strategies/
├── NostalgiaForInfinityX7.py -> ../../NostalgiaForInfinityX7.py  (symlink from upstream)
├── NostalgiaForInfinityX6.py -> ../../NostalgiaForInfinityX6.py  (symlink from upstream)
├── ElliotV5_SMA/                    # Your custom strategies
│   ├── ElliotV5_SMA.py
│   ├── ElliotV5_SMA_Shorts.py
│   ├── trades_jan2026.csv
│   └── analysis.csv
├── NASOSv4/                         # Your custom strategies
│   ├── NASOSv4.py
│   ├── NASOSv4_Shorts.py
│   └── backtest_results.json
└── El/                              # Your custom strategies
    ├── EI4_t4c0s_V2_2.py
    └── EI4_t4c0s_V2_2_Shorts.py
```

**Why this works**:
- Upstream NFI strategies are symlinks (no conflicts during merge)
- Your custom subdirectories are not in upstream (no conflicts)
- Freqtrade loads strategies recursively from subdirectories
- GitHub Actions handles sync automatically

---

## Setup Instructions

### 1. Local Development (Mac) - One-Time Setup

#### Enable GitHub Actions on your fork:
1. Go to: https://github.com/vaskosmihaylov/nfi-custom-strategies/actions
2. Click "I understand my workflows, go ahead and enable them"
3. The workflow will run daily at 6 AM UTC automatically

#### Push changes to fork:
```bash
cd /Users/vasmihay/Personal/AI_Projects/NostalgiaForInfinity

# Add and commit changes
git add -A
git commit -m "Simplify: Remove nfi directory, use upstream symlinks, add GitHub Actions sync"
git push nfi-custom-strategies main
```

#### Manual sync (if needed):
Go to: https://github.com/vaskosmihaylov/nfi-custom-strategies/actions/workflows/sync-upstream.yml
Click "Run workflow"

---

### 2. Production Server (Amazon Ubuntu)

#### Simple pull script:
Create `/opt/trading/nfi-custom-strategies/pull-and-restart.sh`:

```bash
#!/bin/bash
# Simple production deployment script

cd /opt/trading/nfi-custom-strategies || exit 1

echo "[$(date)] Pulling latest changes..."
if git pull origin main; then
    echo "[$(date)] ✅ Pull successful"

    # Show version
    VERSION=$(grep 'return "v' NostalgiaForInfinityX7.py 2>/dev/null | head -1 | sed 's/.*return "\(v[^"]*\)".*/\1/')
    echo "[$(date)] Version: $VERSION"

    # Restart containers
    echo "[$(date)] Restarting docker containers..."
    docker-compose -f docker-compose-multi-strategies.yml restart

    echo "[$(date)] ✅ Deployment complete"
else
    echo "[$(date)] ❌ Pull failed"
    exit 1
fi
```

#### Make executable:
```bash
chmod +x /opt/trading/nfi-custom-strategies/pull-and-restart.sh
```

#### Add to cron (optional):
```bash
crontab -e

# Add this line to auto-pull daily at 7 AM UTC (1 hour after GitHub sync)
0 7 * * * /opt/trading/nfi-custom-strategies/pull-and-restart.sh >> /opt/trading/nfi-custom-strategies/deploy.log 2>&1
```

#### Or run manually when needed:
```bash
cd /opt/trading/nfi-custom-strategies
./pull-and-restart.sh
```

---

## How It Works

### Daily Automation (Hands-Off)

**6:00 AM UTC**:
- GitHub Actions runs on your fork
- Fetches from `iterativv/NostalgiaForInfinity`
- Merges changes automatically
- Auto-resolves conflicts (prefers upstream for NFI files)
- Your custom subdirectories are untouched (no conflicts)

**7:00 AM UTC** (optional if cron enabled):
- Production server pulls from your fork
- Restarts docker containers
- All strategies updated

### Manual Deployment

**When you update your custom strategies**:
```bash
# On local Mac
git add user_data/strategies/YourCustomStrategy/
git commit -m "Update custom strategy"
git push nfi-custom-strategies main

# On production (5 minutes later)
cd /opt/trading/nfi-custom-strategies
./pull-and-restart.sh
```

---

## What Changed

### Deleted (Complexity Removed)
- ❌ `user_data/strategies/nfi/` directory (was causing duplicate class name issues)
- ❌ `daily-sync-nfi.sh` (no longer needed)
- ❌ `daily-sync-and-push.sh` (no longer needed)
- ❌ `sync-nfi-strategy.sh` (no longer needed)
- ❌ `pull-latest-production.sh` (replaced with simpler script)
- ❌ `.git/hooks/post-merge` (no longer needed)
- ❌ `skip-worktree` management (no longer needed)
- ❌ Local Mac cron job (GitHub Actions does this now)

### Added (Simplification)
- ✅ `.github/workflows/sync-upstream.yml` (GitHub handles sync automatically)
- ✅ Simple production script: `pull-and-restart.sh` (just pull + restart)

### Updated
- ✅ `docker-compose-multi-strategies.yml` line 39: `user_data/strategies/nfi` → `user_data/strategies`

---

## Verification

### Check NFI strategy loads correctly:
```bash
# On production
docker logs freqtrade-nfi-x7 | grep -i "nostalgia"

# Should show:
# "Strategy NostalgiaForInfinityX7 loaded successfully"
```

### Check custom strategies load:
```bash
# List available strategies
docker exec freqtrade-elliotv5_sma freqtrade list-strategies --strategy-path user_data/strategies/ElliotV5_SMA
```

### Check GitHub Actions:
Visit: https://github.com/vaskosmihaylov/nfi-custom-strategies/actions

---

## Troubleshooting

### If production shows error after pull:
```bash
# Check what file is being loaded
docker exec freqtrade-nfi-x7 ls -la /freqtrade/user_data/strategies/ | grep X7

# Should show symlink:
# lrwxr-xr-x NostalgiaForInfinityX7.py -> ../../NostalgiaForInfinityX7.py

# Check class name in root file
docker exec freqtrade-nfi-x7 grep 'class Nostalgia' /freqtrade/NostalgiaForInfinityX7.py
# Should show: class NostalgiaForInfinityX7(IStrategy):

# Restart container
docker restart freqtrade-nfi-x7
docker logs freqtrade-nfi-x7 | tail -50
```

### If GitHub Actions fails:
1. Check: https://github.com/vaskosmihaylov/nfi-custom-strategies/actions
2. Click on failed workflow run
3. Review error logs
4. Merge conflicts may require manual intervention (rare)

---

## Summary

### What You Maintain Now
- **Local Mac**: Just develop and push (no cron, no scripts)
- **GitHub**: Syncs automatically daily (no action needed)
- **Production**: One simple script: `pull-and-restart.sh`

### Total Code Maintained
- 1 GitHub Actions workflow (~50 lines)
- 1 production script (~20 lines)
- 1 docker-compose update (1 line changed)

### vs Previous Complexity
- ~~3 bash scripts (250+ lines)~~
- ~~2 cron jobs~~
- ~~skip-worktree management~~
- ~~git hooks~~
- ~~file copying/syncing~~
- ~~timing dependencies~~

**70% reduction in complexity!** 🎉
