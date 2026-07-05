#!/usr/bin/env bash
# Exit immediately on errors, undefined vars, or failed pipeline commands
set -euo pipefail

# Ensure we're working with a clean state
echo "=== Starting data preparation ==="

# Set GIT_LFS_SKIP_SMUDGE to prevent automatic LFS downloads
export GIT_LFS_SKIP_SMUDGE=1

# Clean any existing data_subset directory
rm -rf data_subset
mkdir -p data_subset/data/carpk/labels
mkdir -p data_subset/data/carpk/raw/ImageSets
mkdir -p data_subset/data/carpk/raw/labels/


# Selectively pull ONLY the LFS files we need
echo "=== Pulling selected LFS files ==="
git lfs pull --include="data/carpk/labels/*" --exclude="*"
git lfs pull --include="data/carpk/raw/ImageSets/*" --exclude="*"
git lfs pull --include="data/carpk/raw/images/*" --exclude="*"

# Copy metadata/config files
echo "=== Copying metadata files ==="
cp -v data/carpk/data.yaml data_subset/data/carpk/
cp -v data/carpk/raw/ImageSets/* data_subset/data/carpk/raw/ImageSets/
cp -r data/carpk/raw/labels/* data_subset/data/carpk/raw/labels/


# Archive the subset for Docker (without parent directory)
echo "=== Creating data archive ==="
tar czf data_subset.tar.gz -C data_subset .

# Verify archive size (for debugging)
echo "Archive size: $(du -h data_subset.tar.gz | cut -f1)"
echo "Number of files in archive: $(tar tzf data_subset.tar.gz | wc -l)"

# Build Docker image
echo "=== Building Docker image ==="
docker build -t "$IMAGE_NAME:latest" .

# Push to registry
echo "=== Pushing to registry ==="
echo "$CI_REGISTRY_PASSWORD" | docker login \
    --username "$CI_REGISTRY_USER" --password-stdin "$CI_REGISTRY"
docker push "$IMAGE_NAME:latest"

echo "=== Data preparation complete ==="
