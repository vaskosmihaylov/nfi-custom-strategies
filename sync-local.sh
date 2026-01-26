#!/bin/bash
# Sync local repository with upstream NostalgiaForInfinity

set -e

echo "🔄 Syncing local repository with upstream..."

# Add upstream remote if it doesn't exist
if ! git remote | grep -q '^upstream$'; then
    echo "📍 Adding upstream remote..."
    git remote add upstream https://github.com/iterativv/NostalgiaForInfinity.git
fi

# Fetch upstream changes
echo "⬇️  Fetching upstream changes..."
git fetch upstream

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📌 Current branch: $CURRENT_BRANCH"

# Merge upstream/main into current branch
echo "🔀 Merging upstream/main..."
if ! git merge upstream/main --no-edit; then
    echo "⚠️  Merge conflicts detected, attempting auto-resolution..."

    # For deleted files that were modified upstream, accept upstream version
    for file in $(git diff --name-only --diff-filter=U); do
        echo "  Resolving conflict in: $file"
        if [[ "$file" == NostalgiaForInfinity*.py ]]; then
            echo "    → Taking upstream version"
            git checkout --theirs "$file"
            git add "$file"
        else
            echo "    ⚠️  Manual resolution needed for: $file"
        fi
    done

    # Also handle deleted files (modify/delete conflicts)
    git status --porcelain | grep '^DU' | awk '{print $2}' | while read file; do
        echo "  File deleted locally but modified upstream: $file"
        echo "    → Accepting upstream version"
        git add "$file"
    done

    # Complete the merge
    if git diff --cached --quiet; then
        echo "❌ No changes staged. Cannot complete merge."
        echo "Please resolve conflicts manually."
        exit 1
    fi

    git commit --no-edit
fi

# Push to your fork (nfi-custom-strategies)
echo "⬆️  Pushing to nfi-custom-strategies..."
git push nfi-custom-strategies "$CURRENT_BRANCH"

echo "✅ Sync completed successfully!"
echo "📝 Latest commit: $(git log -1 --oneline)"
