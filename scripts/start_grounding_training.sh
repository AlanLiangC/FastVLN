#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
run_dir="$(realpath -m "${STREAMNAV_RUN_DIR:-runs/pointcot_grounding_refined}")"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
IFS=',' read -r -a grounding_gpus <<< "$CUDA_VISIBLE_DEVICES"
world="${STREAMNAV_WORLD_SIZE:-${#grounding_gpus[@]}}"
exec python tools/launch_job.py --name training --run-dir "$run_dir" --monitor -- \
  -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$world" \
  --module streamnav.training.grounding_trainer --run-dir "$run_dir" "$@"
