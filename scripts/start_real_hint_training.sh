#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
export STREAMNAV_RUN_DIR="${STREAMNAV_RUN_DIR:-runs/finetune_15300}"
export STREAMNAV_TRAIN_CONFIG="${STREAMNAV_TRAIN_CONFIG:-configs/pointcot_15300.yaml}"
exec bash scripts/start_pointcot_training.sh "$@"
