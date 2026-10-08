#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
base_python="${STREAMNAV_BASE_PYTHON:-/root/miniconda3/envs/py312torch210cu126/bin/python}"
if [[ ! -x "$base_python" ]]; then base_python="$(command -v python3.12)"; fi
if [[ ! -x .venv/bin/python ]]; then
  "$base_python" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --index-url https://pypi.org/simple uv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -e '.[dev]'
conda_binary="${STREAMNAV_CONDA:-/root/miniconda3/bin/conda}"
if [[ ! -x runtime/habitat-env/bin/python ]]; then
  "$conda_binary" create -y -p "$STREAMNAV_RUNTIME/habitat-env" --override-channels \
    -c conda-forge -c aihabitat python=3.9 habitat-sim=0.3.3 headless numpy=1.26 pyzmq pillow pyyaml
fi
python -m pip freeze > runtime/learner-installed.txt
"$conda_binary" list -p "$STREAMNAV_RUNTIME/habitat-env" --explicit > runtime/habitat-explicit.txt
echo "Environments ready. Source scripts/env.sh before running commands."
