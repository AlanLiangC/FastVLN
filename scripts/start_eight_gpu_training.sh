#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir="${STREAMNAV_RUN_DIR:-${STREAMNAV_RUN_ROOT:-runs/lightnav_kda_active}}"
extra=()
if [[ "${STREAMNAV_RESUME:-0}" == 1 ]]; then
  test -f "$run_dir/checkpoints/latest/optimizer.pt"
  extra+=(--resume "$run_dir/checkpoints/latest")
fi
STREAMNAV_RUN_DIR="$run_dir" STREAMNAV_WORLD_SIZE=8 \
  bash scripts/start_lightnav_training.sh "${extra[@]}" "$@"
