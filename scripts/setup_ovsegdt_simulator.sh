#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
conda_binary="${STREAMNAV_CONDA:-/root/miniconda3/bin/conda}"
if [[ ! -x runtime/ovsegdt-sim-env/bin/python ]]; then
  "$conda_binary" create -y -p "$STREAMNAV_RUNTIME/ovsegdt-sim-env" \
    --file configs/baselines/ovsegdt-simulator-explicit.txt
fi
runtime/ovsegdt-sim-env/bin/python -c \
  'import habitat_sim; assert habitat_sim.__version__ == "0.2.3", habitat_sim.__version__'
