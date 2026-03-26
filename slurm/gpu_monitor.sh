#!/bin/bash
set -euo pipefail

OUT_CSV="${1:-}"
INTERVAL_SEC="${2:-1.0}"

if [ -z "${OUT_CSV}" ]; then
  echo "Usage: $0 <output_csv> [interval_sec]"
  exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"
echo "ts,gpu_index,power_w,util_gpu_pct,util_mem_pct,mem_used_mb,temp_c" > "${OUT_CSV}"

running=1
trap 'running=0' INT TERM

while [ "${running}" -eq 1 ]; do
  ts="$(date +%s.%3N)"
  if line="$(nvidia-smi --query-gpu=index,power.draw,utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1)"; then
    if [ -n "${line}" ]; then
      echo "${ts},${line//, /,}" >> "${OUT_CSV}"
    fi
  fi
  sleep "${INTERVAL_SEC}" || true
done
