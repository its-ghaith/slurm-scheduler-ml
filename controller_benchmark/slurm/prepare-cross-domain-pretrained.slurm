#!/bin/bash
#SBATCH --job-name=rapec-pretrain
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/logs/pretrain-cross-domain-%j.out
#SBATCH --error=/workspace/logs/pretrain-cross-domain-%j.err

set -euo pipefail

: "${BENCHMARK_WORKSPACE:?BENCHMARK_WORKSPACE is required}"
: "${BENCHMARK_MANIFEST:?BENCHMARK_MANIFEST is required}"

cd "$BENCHMARK_WORKSPACE"
export PYTHONPATH="$BENCHMARK_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export CONTROLLER_DATASET_ROOT="${CONTROLLER_DATASET_ROOT:-/workspace-cache/controller-datasets}"

python -m controller_benchmark.prepare_cross_domain_pretrained \
  --manifest "$BENCHMARK_MANIFEST" \
  --output-root "${PRETRAINING_OUTPUT_ROOT:-/workspace-cache/controller-pretrained-cross-domain}" \
  --pretraining-checkpoints "${PRETRAINING_CHECKPOINTS:-20}"
