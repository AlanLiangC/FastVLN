#!/usr/bin/env bash
set -euo pipefail
STREAMNAV_WORLD_SIZE="${STREAMNAV_WORLD_SIZE:-1}" \
  exec bash "$(dirname "${BASH_SOURCE[0]}")/start_lightnav_training.sh" "$@"
