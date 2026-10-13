#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
run_dir="$(realpath -m "${STREAMNAV_RUN_DIR:-runs/qwen35_pointcot_active}")"
world="${STREAMNAV_WORLD_SIZE:-6}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
if [[ $# -eq 0 ]]; then
  init="${STREAMNAV_INIT_CHECKPOINT:-runs/pointcot_grounding_refined/checkpoints/best}"
  if [[ ! -f "$init/adapters.safetensors" ]]; then
    echo "Run grounding first; checkpoint missing: $init" >&2
    exit 1
  fi
  set -- --initialize-from "$init"
fi
exec python tools/launch_job.py --name training --run-dir "$run_dir" --monitor -- \
  -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$world" \
  --module streamnav.training.pointcot_trainer --run-dir "$run_dir" "$@"
