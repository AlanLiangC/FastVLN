#!/usr/bin/env bash
# Source this for OVSegDT only; the FastVLN interpreter remains separate.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export YOLO_CONFIG_DIR="$STREAMNAV_RUNTIME/cache/ovsegdt/ultralytics"
export MPLCONFIGDIR="$STREAMNAV_RUNTIME/cache/ovsegdt/matplotlib"
export OVSEGDT_PYTHON="$STREAMNAV_RUNTIME/ovsegdt-env/bin/python"
export PYTHONPATH="$STREAMNAV_ROOT/tools"
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
