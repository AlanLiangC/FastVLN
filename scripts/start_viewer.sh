#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
viewer_checkpoint="${1:-checkpoints/fastvln_15300}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ ! -f "$viewer_checkpoint/adapters.safetensors" ]]; then
  echo "A trained checkpoint is required: $viewer_checkpoint" >&2
  exit 1
fi
viewer_gpu="${STREAMNAV_VIEWER_GPU:-3}"
viewer_run_dir="${STREAMNAV_VIEWER_RUN_DIR:-runs/evaluation_15300}"
exec python -m streamnav.serving.server "checkpoint=$viewer_checkpoint" "run_dir=$viewer_run_dir" "device=cuda:$viewer_gpu" "habitat.gpu_device_id=$viewer_gpu" "$@"
