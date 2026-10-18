#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
run_dir="$(realpath -m "${STREAMNAV_RUN_DIR:-runs/qwen35_pointcot_opos_hint}")"
world="${STREAMNAV_WORLD_SIZE:-8}"
if [[ ! "$world" =~ ^[1-9][0-9]*$ ]]; then
  echo "STREAMNAV_WORLD_SIZE must be a positive integer" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 "$((world - 1))")}"
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
