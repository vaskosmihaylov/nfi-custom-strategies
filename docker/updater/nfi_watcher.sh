#!/bin/bash

EXCHANGE="${EXCHANGE:-binance}"
WATCH_URL="https://raw.githubusercontent.com/iterativv/NostalgiaForInfinity/main/configs/blacklist-${EXCHANGE}.json"
LAST_ETAG_FILE="/tmp/nfi_blacklist_etag"
UPDATE_SCRIPT="/scripts/update_nfi.sh"

echo "NFI Watcher running."
echo "Watching : $WATCH_URL"
echo "Interval : 60 seconds"

while true; do
    # Fetch only the HTTP headers (no file download) — lightweight on bandwidth and GitHub rate limits
    CURRENT_ETAG=$(curl -sI "$WATCH_URL" | grep -i "etag" | awk '{print $2}' | tr -d '\r')

    if [ -z "$CURRENT_ETAG" ]; then
        echo "$(date): Warning: could not fetch ETag. Retrying in 60s..."
        sleep 60
        continue
    fi

    # First run — initialise the stored ETag
    if [ ! -f "$LAST_ETAG_FILE" ]; then
        echo "$CURRENT_ETAG" > "$LAST_ETAG_FILE"
        echo "$(date): Watcher initialised. ETag: $CURRENT_ETAG"
    fi

    LAST_ETAG=$(cat "$LAST_ETAG_FILE")

    if [ "$CURRENT_ETAG" != "$LAST_ETAG" ]; then
        echo "==================================================="
        echo "$(date): Blacklist change detected!"
        echo "  Old ETag: $LAST_ETAG"
        echo "  New ETag: $CURRENT_ETAG"
        echo "Triggering update..."

        bash "$UPDATE_SCRIPT"

        # Save new ETag only after a successful update run
        echo "$CURRENT_ETAG" > "$LAST_ETAG_FILE"
        echo "$(date): Update complete. Back to watch mode."
        echo "==================================================="
    fi

    sleep 60
done
