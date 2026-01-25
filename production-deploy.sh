#!/bin/bash
# Simple Production Deployment Script
# Copy this to: /opt/trading/nfi-custom-strategies/pull-and-restart.sh

cd /opt/trading/nfi-custom-strategies || {
    echo "[$(date)] ❌ ERROR: Could not cd to /opt/trading/nfi-custom-strategies"
    exit 1
}

echo "[$(date)] 📥 Pulling latest changes from fork..."

if git pull origin main; then
    echo "[$(date)] ✅ Pull successful"

    # Show NFI version
    if [ -f "NostalgiaForInfinityX7.py" ]; then
        VERSION=$(grep 'return "v' NostalgiaForInfinityX7.py 2>/dev/null | head -1 | sed 's/.*return "\(v[^"]*\)".*/\1/')
        CLASS=$(grep 'class Nostalgia.*X7' NostalgiaForInfinityX7.py 2>/dev/null | head -1 | sed 's/class \([^(]*\).*/\1/')
        echo "[$(date)] 📊 NFI X7 Version: $VERSION"
        echo "[$(date)] 🔍 Class Name: $CLASS"
    fi

    # Restart all containers
    echo "[$(date)] 🔄 Restarting docker containers..."
    if docker-compose -f docker-compose-multi-strategies.yml restart; then
        echo "[$(date)] ✅ Containers restarted successfully"

        # Wait and check for errors
        sleep 10
        ERROR_COUNT=$(docker-compose -f docker-compose-multi-strategies.yml logs --tail=50 2>&1 | grep -i "impossible to load\|error.*strategy" | wc -l)

        if [ "$ERROR_COUNT" -gt 0 ]; then
            echo "[$(date)] ⚠️  WARNING: Found errors in container logs"
            docker-compose -f docker-compose-multi-strategies.yml logs --tail=20
        else
            echo "[$(date)] ✅ No errors detected"
        fi

        echo "[$(date)] ✅ Deployment complete"
    else
        echo "[$(date)] ❌ ERROR: Container restart failed"
        exit 1
    fi
else
    echo "[$(date)] ❌ ERROR: Git pull failed"
    exit 1
fi
