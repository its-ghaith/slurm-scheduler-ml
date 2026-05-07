#!/usr/bin/env bash

# Exit immediately on errors, undefined vars, or failed pipeline commands
set -euo pipefail

# Collect LFS-tracked files
LFS_FILES=$(git lfs ls-files | awk '{print $NF}')

echo "Checking synthetic_cells/raw..."
# 1. Check raw/ has exactly 200 LFS files
RAW_COUNT=$(echo "$LFS_FILES" | grep -c "^data/synthetic_cells/raw/")
echo "Found $RAW_COUNT LFS files in synthetic_cells/raw/"
test "$RAW_COUNT" -eq 400 || (echo "Expected 400 LFS files in synthetic_cells/raw/, got $RAW_COUNT" && exit 1)

# 2. Check synthetic_cells/data.yaml exists
test -f data/synthetic_cells/data.yaml || (echo "data/synthetic_cells/data.yaml missing!" && exit 1)
echo "synthetic_cells OK"
echo "Checking uc_cells..."
# 3. Check uc_cells/data.yaml exists
test -f data/uc_cells/data.yaml || (echo "data/uc_cells/data.yaml missing!" && exit 1)
# 4. Check uc_cells/images/test exists
test -d data/uc_cells/raw || (echo "Missing directory data/uc_cells/raw" && exit 1);
RAW_COUNT=$(echo "$LFS_FILES" | grep -c "^data/uc_cells/raw/labels_xywh")
# test "$RAW_COUNT" -eq 249 || (echo "Expected 249 label files in uc_cells/raw/labels_xywh, got $RAW_COUNT" && exit 1)
echo "uc_cells OK"

echo "Checking carpk..."
test -f data/carpk/data.yaml || (echo "data/carpk/data.yaml missing!" && exit 1)
test -d data/carpk/raw || (echo "Missing directory data/carpk/raw" && exit 1);
RAW_COUNT=$(echo "$LFS_FILES" | grep -c "^data/carpk/raw/images")
test "$RAW_COUNT" -eq 1448 || (echo "Expected 1,448 LFS files in carpk/raw/images, got $RAW_COUNT" && exit 1)
echo "carpk OK"
