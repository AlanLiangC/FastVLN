#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
exec python tools/launch_job.py --name pipeline --run-dir runs/pointcot_pipeline -- \
  -m streamnav.training.pipeline --world-size "${STREAMNAV_WORLD_SIZE:-1}" "$@"
