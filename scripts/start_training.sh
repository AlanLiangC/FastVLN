#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
run_dir="${STREAMNAV_RUN_DIR:-runs/streamnav_single_gpu}"
mkdir -p "$run_dir"
keepalive_args=()
learner_gpu=0
for argument in "$@"; do
  if [[ "$argument" == device=cuda:* ]]; then learner_gpu="${argument##*:}"; fi
done
if [[ "${STREAMNAV_KEEPALIVE:-1}" == 1 ]]; then
  keepalive_args=(--keepalive-gpu "$learner_gpu")
fi
exec python tools/launch_job.py --name training --run-dir "$run_dir" --monitor "${keepalive_args[@]}" -- \
  -m streamnav.training.trainer "run_dir=$run_dir" "habitat.gpu_device_id=$learner_gpu" "$@"
