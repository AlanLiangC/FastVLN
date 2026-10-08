#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root="${STREAMNAV_RUN_ROOT:-runs/streamnav_active}"
# One reference-recipe experiment uses all eight GPUs.
if [[ "${STREAMNAV_RESUME:-0}" == 1 ]]; then
  test -f "$root/ealm/checkpoints/latest/optimizer.pt"
fi
extra=()
if [[ "${STREAMNAV_RESUME:-0}" == 1 ]]; then
  extra+=("checkpoint=$root/ealm/checkpoints/latest")
fi
STREAMNAV_RUN_DIR="$root/ealm" STREAMNAV_WORLD_SIZE=8 STREAMNAV_GPU_OFFSET=0 \
  bash scripts/start_distributed_training.sh "${extra[@]}" "$@"
