#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
run_dir="$(realpath -m "${STREAMNAV_RUN_DIR:-runs/streamnav_active/ealm}")"
world="${STREAMNAV_WORLD_SIZE:-4}"
offset="${STREAMNAV_GPU_OFFSET:-0}"
keepalive_args=()
if [[ "${STREAMNAV_KEEPALIVE:-1}" == 1 ]]; then
  for ((i=offset; i<offset+world; i++)); do keepalive_args+=(--keepalive-gpu "$i"); done
fi
exec python tools/launch_job.py --name training --run-dir "$run_dir" --monitor "${keepalive_args[@]}" -- \
  -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$world" \
  --module streamnav.training.trainer "run_dir=$run_dir" "distributed.gpu_offset=$offset" "$@"
