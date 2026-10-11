#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
run_dir="$(realpath -m "${STREAMNAV_RUN_DIR:-runs/lightnav_kda_active}")"
world="${STREAMNAV_WORLD_SIZE:-8}"
exec python tools/launch_job.py --name training --run-dir "$run_dir" --monitor -- \
  -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$world" \
  --module streamnav.training.lightnav_trainer --run-dir "$run_dir" "$@"
