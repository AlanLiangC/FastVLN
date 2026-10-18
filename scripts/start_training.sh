#!/usr/bin/env bash
set -euo pipefail
STREAMNAV_WORLD_SIZE="${STREAMNAV_WORLD_SIZE:-8}" \
  exec bash "$(dirname "${BASH_SOURCE[0]}")/start_pointcot_training.sh" "$@"
