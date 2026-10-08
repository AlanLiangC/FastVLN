#!/usr/bin/env bash
# Source this file from any directory. No cache is written to /root.
export STREAMNAV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export STREAMNAV_RUNTIME="${STREAMNAV_RUNTIME:-$STREAMNAV_ROOT/runtime}"
export HF_HOME="$STREAMNAV_RUNTIME/cache/huggingface"
export PIP_CACHE_DIR="$STREAMNAV_RUNTIME/cache/pip"
export UV_CACHE_DIR="$STREAMNAV_RUNTIME/cache/uv"
export CONDA_PKGS_DIRS="$STREAMNAV_RUNTIME/cache/conda"
export XDG_CACHE_HOME="$STREAMNAV_RUNTIME/cache"
export TRITON_CACHE_DIR="$STREAMNAV_RUNTIME/cache/triton"
export TORCH_HOME="$STREAMNAV_RUNTIME/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$STREAMNAV_RUNTIME/cache/torchinductor"
export TMPDIR="$STREAMNAV_RUNTIME/tmp"
export PYTHONPATH="$STREAMNAV_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONUNBUFFERED=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PATH="$STREAMNAV_ROOT/.venv/bin:$PATH"
mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$STREAMNAV_RUNTIME/logs"
