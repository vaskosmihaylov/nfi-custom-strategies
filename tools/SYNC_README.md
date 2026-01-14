# Repository Sync Scripts

## Production Server (Amazon EC2) - AUTOMATED

**Script**: `update_upstream_production.sh`

**What it does**: Syncs production with upstream NFI hourly, restarts affected containers

**Setup on EC2**:
```bash
# 1. Edit configuration (update paths if needed)
nano /opt/trading/nfi-custom-strategies/tools/update_upstream_production.sh

# 2. Make executable
chmod +x /opt/trading/nfi-custom-strategies/tools/update_upstream_production.sh

# 3. Test it
./tools/update_upstream_production.sh

# 4. Add to cron (runs every hour at :05)
crontab -e
# Add: 5 * * * * /opt/trading/nfi-custom-strategies/tools/update_upstream_production.sh
```

**Key config**:
- `REPO_PATH="/opt/trading/nfi-custom-strategies"`
- `RESTART_STRATEGY="auto"` (only affected containers restart)
- `CONFLICT_STRATEGY="abort"` (safe for production)
- Logs: `/var/log/nfi-update.log`

---

## Local Mac - MANUAL

**Script**: `sync_from_upstream.sh`

**What it does**: Pulls upstream NFI changes to your Mac, pushes to your fork

**Usage**:
```bash
# Get latest from upstream NFI
./tools/sync_from_upstream.sh

# Or just use git directly
git pull origin main
git push nfi-custom-strategies main
```

**When to run**: Whenever you want latest upstream changes (new strategies, blacklist updates, etc.)

---

## Your Git Setup

**Mac**:
```
origin                  → iterativv/NostalgiaForInfinity (upstream)
nfi-custom-strategies   → vaskosmihaylov/nfi-custom-strategies (your fork)
```

**Production**:
```
origin    → vaskosmihaylov/nfi-custom-strategies (your fork)
upstream  → iterativv/NostalgiaForInfinity (auto-added by script)
```

---

## Quick Commands

```bash
# Production: Check logs
ssh ubuntu@your-ec2 "tail -f /var/log/nfi-update.log"

# Production: Manual sync
ssh ubuntu@your-ec2 "cd /opt/trading/nfi-custom-strategies && ./tools/update_upstream_production.sh"

# Mac: Get upstream changes
./tools/sync_from_upstream.sh

# Mac: See what's new upstream
git fetch origin main && git log --oneline HEAD..origin/main
```

That's it!
