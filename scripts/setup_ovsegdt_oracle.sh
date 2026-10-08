#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
oracle_dir=runtime/vendor/frontier_exploration
oracle_commit=a8890d68cfa0d10254238abe9266a76856cb1f17
if [[ ! -d "$oracle_dir/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/naokiyokoyama/frontier_exploration.git "$oracle_dir"
fi
git -C "$oracle_dir" checkout "$oracle_commit"
runtime/habitat-env/bin/python -m pip install --index-url https://pypi.org/simple \
  'gym==0.23.0' 'hydra-core==1.3.2' 'omegaconf==2.3.1' \
  'opencv-python-headless==4.11.0.86' 'numpy<2' 'pandas<3' \
  'yacs==0.1.8' 'imageio-ffmpeg<1' 'networkx<4' 'scikit-image<0.25' imageio attrs tqdm
runtime/habitat-env/bin/python -m pip freeze > runtime/oracle-installed.txt
