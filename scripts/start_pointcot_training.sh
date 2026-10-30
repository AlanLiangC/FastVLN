#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
run_dir="$(realpath -m "${STREAMNAV_RUN_DIR:-runs/finetune_15300}")"
world="${STREAMNAV_WORLD_SIZE:-8}"
if [[ ! "$world" =~ ^[1-9][0-9]*$ ]]; then
  echo "STREAMNAV_WORLD_SIZE must be a positive integer" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 "$((world - 1))")}"
explicit_restore=false
for argument in "$@"; do
  case "$argument" in
    --resume|--resume=*|--fork-from|--fork-from=*|--initialize-from|--initialize-from=*)
      explicit_restore=true ;;
  esac
done
if [[ "$explicit_restore" == false ]]; then
  checkpoint="checkpoints/fastvln_15300"
  if [[ -f "$run_dir/checkpoints/latest/manifest.json" ]]; then
    checkpoint="$run_dir/checkpoints/latest"
  fi
  if [[ ! -f "$checkpoint/optimizer.pt" ]]; then
    echo "Full navigation training state missing: $checkpoint" >&2
    exit 1
  fi
  set -- --resume "$checkpoint" "$@"
fi
exec python tools/launch_job.py --name training --run-dir "$run_dir" --monitor -- \
  -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$world" \
  --module streamnav.training.pointcot_trainer --run-dir "$run_dir" \
  --config "${STREAMNAV_TRAIN_CONFIG:-configs/pointcot_15300.yaml}" "$@"
