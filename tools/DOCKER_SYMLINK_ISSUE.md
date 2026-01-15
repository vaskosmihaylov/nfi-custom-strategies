# Docker + Python Import + Symlinks Issue

## Problem Summary

Docker containers **cannot reliably import Python modules through symlinks**, even though the symlinks are valid and accessible within the container.

## Symptoms

```bash
# Symlink exists and is valid
$ ls -lh /freqtrade/user_data/strategies/nfi/NostalgiaForInfinityX7.py
lrwxrwxrwx 1 ftuser ftuser 34 Jan 15 17:56 NostalgiaForInfinityX7.py -> ../../../NostalgiaForInfinityX7.py

# File is accessible
$ cat /freqtrade/user_data/strategies/nfi/NostalgiaForInfinityX7.py
# [file contents displayed successfully]

# But Python import FAILS
$ python3 -c "from NostalgiaForInfinityX7 import NostalgiaForInfinityX7"
ImportError: No module named 'NostalgiaForInfinityX7'

# Freqtrade error
2026-01-15 18:01:02,620 - freqtrade - ERROR - Impossible to load Strategy 'NostalgiaForInfinityX7'.
This class does not exist or contains Python code errors.
```

## Root Cause

Python's module import system doesn't properly follow symlinks in Docker mounted volumes. This is a known limitation when:
- Files are symlinked across Docker volume boundaries
- Python tries to resolve the module path for imports
- The symlink target is outside the immediate module search path

## Solution

**Replace symlinks with actual file copies.**

### Step 1: One-Time Setup (Already Done)

On your production server, you've already:
```bash
cd /opt/trading/nfi-custom-strategies
rm user_data/strategies/nfi/NostalgiaForInfinityX7.py  # Remove symlink
cp NostalgiaForInfinityX7.py user_data/strategies/nfi/  # Copy actual file
./deploy-multi-strategies.sh restart nfi-x7
```

### Step 2: Automated Future Updates (Implemented)

The `update_upstream_production.sh` script has been updated to automatically:

1. **Detect changed strategy files** after upstream merge
2. **Copy files to Docker paths** (new `copy_strategies_to_docker_paths()` function)
3. **Remove any symlinks** before copying
4. **Restart affected containers**

#### Code Changes

Added new function at line 348:
```bash
copy_strategies_to_docker_paths() {
    local changed_files="$1"

    # For NostalgiaForInfinityX7.py specifically:
    # 1. Remove symlink if exists
    # 2. Copy actual file from root to user_data/strategies/nfi/
    # 3. Verify copy succeeded
}
```

Called in main() after successful merge (line 514):
```bash
# Update was successful
log_success "Successfully merged upstream changes"

# Copy strategy files to Docker-accessible paths
if [ -n "$CHANGED_FILES" ]; then
    copy_strategies_to_docker_paths "$CHANGED_FILES"
fi

# Detect affected strategies
# ... rest of restart logic
```

## Diagnostic Tools

Three scripts created for troubleshooting:

### 1. `diagnose-strategy-load.sh`
Comprehensive diagnostic tool that checks:
- File locations (main file, symlink, Docker container)
- Python syntax
- Class definition
- Import errors in container
- Recent changes
- Container logs

Usage:
```bash
./tools/diagnose-strategy-load.sh [strategy-name]
./tools/diagnose-strategy-load.sh NostalgiaForInfinityX7
```

### 2. `fix-nfi-x7-symlink.sh` (Deprecated)
Originally created to fix symlinks, but symlinks don't work.
Kept for reference only. Use `setup-nfi-x7-docker.sh` instead.

### 3. `setup-nfi-x7-docker.sh` (Recommended)
One-time setup script that:
- Removes symlinks
- Copies actual files
- Validates Python syntax
- Provides restart instructions

Usage:
```bash
cd /opt/trading/nfi-custom-strategies
./tools/setup-nfi-x7-docker.sh
```

## File Structure

### Before (Broken)
```
/opt/trading/nfi-custom-strategies/
├── NostalgiaForInfinityX7.py                    # Main file
└── user_data/strategies/nfi/
    └── NostalgiaForInfinityX7.py -> ../../../   # Symlink (DOESN'T WORK)
```

### After (Working)
```
/opt/trading/nfi-custom-strategies/
├── NostalgiaForInfinityX7.py                    # Main file (source)
└── user_data/strategies/nfi/
    └── NostalgiaForInfinityX7.py                # Copied file (WORKS)
```

## Testing

Verify the fix:
```bash
# 1. Check file is not a symlink
cd /opt/trading/nfi-custom-strategies
ls -lh user_data/strategies/nfi/NostalgiaForInfinityX7.py
# Should show regular file, NOT symlink

# 2. Test import in container
docker exec freqtrade-nfi-x7 python3 -c \
  "import sys; sys.path.insert(0, '/freqtrade/user_data/strategies/nfi'); \
   from NostalgiaForInfinityX7 import NostalgiaForInfinityX7; \
   print('✓ Import successful!')"

# 3. Check strategy loads in Freqtrade
./deploy-multi-strategies.sh restart nfi-x7
./deploy-multi-strategies.sh logs nfi-x7 | grep "Using resolved strategy"
# Should show: "Using resolved strategy NostalgiaForInfinityX7 from '/freqtrade/user_data/strategies/nfi/NostalgiaForInfinityX7.py'"
```

## Future Considerations

If other strategies need similar treatment, update the `copy_strategies_to_docker_paths()` function to handle them:

```bash
# Add to the if/elif chain:
elif [ "$filename" = "AnotherStrategy.py" ]; then
    # Copy to appropriate location
    cp "$source_file" "${REPO_PATH}/user_data/strategies/another/${filename}"
fi
```

## References

- Docker documentation on volumes: https://docs.docker.com/storage/volumes/
- Python import system: https://docs.python.org/3/reference/import.html
- Related GitHub issues:
  - https://github.com/docker/for-linux/issues/188
  - https://github.com/moby/moby/issues/37965

## Summary

✅ **Problem**: Symlinks + Docker + Python imports = Broken
✅ **Solution**: Copy files instead of symlinking
✅ **Implementation**: Automated in `update_upstream_production.sh`
✅ **Status**: Fixed and tested on production server
