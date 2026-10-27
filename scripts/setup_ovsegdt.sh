#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/ovsegdt_env.sh"
cd "$STREAMNAV_ROOT"

# Native Habitat-Sim/Magnum remain in their existing environment. Only the new
# inference environment gets the author's old Torch and Transformers versions.
if [[ ! -x runtime/habitat-env/bin/python ]]; then
  echo "Requires the existing runtime/habitat-env native simulator installation." >&2
  exit 1
fi
if [[ ! -x "$OVSEGDT_PYTHON" ]]; then
  uv venv --python runtime/habitat-env/bin/python --system-site-packages runtime/ovsegdt-env
fi
uv pip install --python "$OVSEGDT_PYTHON" --index-url https://pypi.org/simple \
  -r configs/baselines/ovsegdt-requirements.txt

patch_file="$STREAMNAV_ROOT/scripts/patches/ovsegdt_inference.patch"
if git -C third_party/OVSegDT apply --check "$patch_file" 2>/dev/null; then
  git -C third_party/OVSegDT apply "$patch_file"
elif ! git -C third_party/OVSegDT apply --reverse --check "$patch_file" 2>/dev/null; then
  echo "OVSegDT auxiliary-loss source differs from the recorded inference patch." >&2
  exit 1
fi
llama_dir="$("$OVSEGDT_PYTHON" -c 'import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent / "models/llama")')"
cp third_party/OVSegDT/modeling_llama.py "$llama_dir/modeling_llama.py"

# Downloading runs in the main interpreter; it does not install packages there.
python tools/download_ovsegdt_weights.py
mkdir -p third_party/OVSegDT/data/text_embeddings
ln -sfn ../../../../checkpoints/ovsegdt/siglip.pkl third_party/OVSegDT/data/text_embeddings/siglip.pkl
if [[ ! -f checkpoints/ovsegdt/siglip.pkl ]]; then
  "$OVSEGDT_PYTHON" tools/prepare_ovsegdt_text.py
fi
"$OVSEGDT_PYTHON" tools/ovsegdt_runtime.py
