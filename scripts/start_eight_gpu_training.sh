#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root="${STREAMNAV_RUN_ROOT:-runs/streamnav_active}"
# Validate both groups before starting either one.
if [[ "${STREAMNAV_RESUME:-0}" == 1 ]]; then
  for name in ealm fixed_mix; do
    test -f "$root/$name/checkpoints/latest/optimizer.pt"
  done
fi
for name in ealm fixed_mix; do
  offset=0
  extra=()
  if [[ "$name" == fixed_mix ]]; then
    offset=4
    extra+=(trainer.ealm.enabled=false)
  fi
  if [[ "${STREAMNAV_RESUME:-0}" == 1 ]]; then
    extra+=("checkpoint=$root/$name/checkpoints/latest")
  fi
  STREAMNAV_RUN_DIR="$root/$name" STREAMNAV_WORLD_SIZE=4 STREAMNAV_GPU_OFFSET="$offset" \
    bash scripts/start_distributed_training.sh "${extra[@]}" "$@"
done
