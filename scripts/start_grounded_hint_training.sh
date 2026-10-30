#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export STREAMNAV_RUN_DIR="${STREAMNAV_RUN_DIR:-runs/finetune_15300_grounded_hints}"
export STREAMNAV_TRAIN_CONFIG="${STREAMNAV_TRAIN_CONFIG:-configs/pointcot_grounded_hints.yaml}"
restore=(--fork-from checkpoints/fastvln_15300 --keep-schedule --skip-initial-eval)
if [[ -f "$STREAMNAV_RUN_DIR/checkpoints/latest/optimizer.pt" ]]; then
  restore=(--resume "$STREAMNAV_RUN_DIR/checkpoints/latest")
fi
exec bash scripts/start_eight_gpu_training.sh "${restore[@]}" "$@"
