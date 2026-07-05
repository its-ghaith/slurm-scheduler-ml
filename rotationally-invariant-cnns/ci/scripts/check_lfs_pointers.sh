#!/usr/bin/env bash

# Exit immediately on errors, undefined vars, or failed pipeline commands
set -euo pipefail

# Collect LFS-tracked files
LFS_FILES=$(git lfs ls-files | awk '{print $NF}')

echo "Checking carpk..."
test -f data/carpk/data.yaml || (echo "data/carpk/data.yaml missing!" && exit 1)
test -d data/carpk/raw || (echo "Missing directory data/carpk/raw" && exit 1);
RAW_COUNT=$(echo "$LFS_FILES" | grep -c "^data/carpk/raw/images")
test "$RAW_COUNT" -eq 1448 || (echo "Expected 1,448 LFS files in carpk/raw/images, got $RAW_COUNT" && exit 1)
echo "carpk OK"
