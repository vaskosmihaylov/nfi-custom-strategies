# Production Server Update - Critical Fixes

## What Was Fixed

### 1. Docker Symlink Issue ❌ → ✅
**Error**:
```
Impossible to load Strategy 'NostalgiaForInfinityX7'. This class does not exist
```

**Root Cause**:
- Symlink `user_data/strategies/NostalgiaForInfinityX7.py -> ../../NostalgiaForInfinityX7.py`
- Docker wasn't mounting root-level files
- Symlink target didn't exist inside container

**Fixed**: Added volume mounts for root-level NFI strategy files to docker-compose

### 2. docker-compose Command Not Found ❌ → ✅
**Error**:
```
docker-compose: command not found
```

**Root Cause**: Modern Docker uses `docker compose` (space) not `docker-compose` (hyphen)

**Fixed**: Updated production script to use correct command

---

## Deploy the Fixes to Production

### Step 1: Pull Latest Changes

```bash
ssh ubuntu@YOUR_SERVER_IP
cd /opt/trading/nfi-custom-strategies

# Pull the fixes
git pull origin main
```

Expected output:
```
From https://github.com/vaskosmihaylov/nfi-custom-strategies
 * branch            main       -> FETCH_HEAD
Updating 11606f13..a3d0a4e7
Fast-forward
 DEPLOYMENT.md                      | 45 +++++++++++++++-
 docker-compose-multi-strategies.yml|  4 ++
 production-deploy.sh               | 65 +++++++++++++----------
 3 files changed, 91 insertions(+), 19 deletions(-)
```

### Step 2: Copy Updated Production Script

```bash
# Make sure the script exists and is executable
ls -la production-deploy.sh
chmod +x production-deploy.sh

# Copy it to the expected location (if not already there)
cp production-deploy.sh pull-and-restart.sh
chmod +x pull-and-restart.sh
```

### Step 3: Restart Containers with New Config

**IMPORTANT**: Use `--force-recreate` to apply the new volume mounts:

```bash
# Stop all containers
docker compose -f docker-compose-multi-strategies.yml down

# Start with new volume mounts (force recreate)
docker compose -f docker-compose-multi-strategies.yml up -d --force-recreate
```

Or use the new script:
```bash
./pull-and-restart.sh
```

Expected output:
```
[INFO] Pulling latest changes from fork...
[SUCCESS] Pull successful
[INFO] NFI X7 Version: v17.3.408
[INFO] Class Name: NostalgiaForInfinityX7
[INFO] Restarting docker containers (reloading configs)...
[SUCCESS] Containers restarted successfully
[INFO] Waiting for containers to start...
[SUCCESS] No errors detected in container logs
[INFO] Running containers:
NAME                          STATUS    PORTS
freqtrade-nfi-x7              running   127.0.0.1:8080->8080/tcp
freqtrade-elliotv5_sma        running   127.0.0.1:8091->8080/tcp
...
[SUCCESS] Deployment complete
```

### Step 4: Verify NFI Strategy Loads

```bash
# Check container logs for NFI-X7
docker logs freqtrade-nfi-x7 | grep -i "nostalgia\|strategy"
```

**Should see**:
```
Strategy NostalgiaForInfinityX7 loaded successfully
```

**Should NOT see**:
```
Impossible to load Strategy 'NostalgiaForInfinityX7'
```

### Step 5: Verify Symlink Works in Container

```bash
# Check symlink exists
docker exec freqtrade-nfi-x7 ls -la /freqtrade/user_data/strategies/NostalgiaForInfinityX7.py

# Should show:
# lrwxr-xr-x 1 ftuser ftuser 31 Jan 25 14:00 NostalgiaForInfinityX7.py -> ../../NostalgiaForInfinityX7.py

# Check symlink resolves (target file exists)
docker exec freqtrade-nfi-x7 ls -la /freqtrade/NostalgiaForInfinityX7.py

# Should show the actual file (not "No such file")

# Verify class name through symlink
docker exec freqtrade-nfi-x7 grep 'class Nostalgia.*X7' /freqtrade/user_data/strategies/NostalgiaForInfinityX7.py

# Should show:
# class NostalgiaForInfinityX7(IStrategy):
```

### Step 6: Check All Strategies Running

```bash
# Show all running containers
docker compose -f docker-compose-multi-strategies.yml ps

# Or use the deploy script
./deploy-multi-strategies.sh status
```

---

## What Changed in docker-compose-multi-strategies.yml

**Added these volume mounts**:
```yaml
volumes:
  - "./user_data:/freqtrade/user_data"
  - "./user_data/data:/freqtrade/user_data/data"
  - "./configs:/freqtrade/configs"
  # NEW: Mount root-level strategy files (needed for symlinks)
  - "./NostalgiaForInfinityX7.py:/freqtrade/NostalgiaForInfinityX7.py:ro"
  - "./NostalgiaForInfinityX6.py:/freqtrade/NostalgiaForInfinityX6.py:ro"
  - "./NostalgiaForInfinityX5.py:/freqtrade/NostalgiaForInfinityX5.py:ro"
```

**Why**: These mounts make the symlinks work by ensuring the target files exist in the container.

---

## Troubleshooting

### If NFI still shows "Impossible to load" error:

```bash
# 1. Check if root file is mounted
docker exec freqtrade-nfi-x7 ls -la /freqtrade/NostalgiaForInfinityX7.py

# If "No such file", volumes aren't mounted correctly
# Solution: Recreate containers
docker compose -f docker-compose-multi-strategies.yml up -d --force-recreate freqtrade-nfi-x7

# 2. Check class name in root file
docker exec freqtrade-nfi-x7 grep 'class Nostalgia' /freqtrade/NostalgiaForInfinityX7.py

# Should show: class NostalgiaForInfinityX7(IStrategy):

# 3. Check logs for other errors
docker logs freqtrade-nfi-x7 --tail 100
```

### If docker compose command not found:

```bash
# Check Docker version
docker --version

# If old Docker, install Docker Compose V2 plugin
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Or use the old command (temporary workaround)
docker-compose -f docker-compose-multi-strategies.yml up -d --force-recreate
```

### If pull-and-restart.sh doesn't exist:

```bash
# It should be tracked in git now
git status
git pull origin main

# If still missing, copy from production-deploy.sh
cp production-deploy.sh pull-and-restart.sh
chmod +x pull-and-restart.sh
```

---

## Summary

**Before**: Symlinks broken, wrong docker command
**After**: Everything works, simplified deployment

**Key Changes**:
1. ✅ Root-level NFI files now mounted in Docker
2. ✅ Symlinks work correctly in containers
3. ✅ Production script uses modern `docker compose` command
4. ✅ Better error checking and status reporting

**Test it**:
```bash
cd /opt/trading/nfi-custom-strategies
./pull-and-restart.sh
```

Should complete without errors and show all containers running! 🎉
