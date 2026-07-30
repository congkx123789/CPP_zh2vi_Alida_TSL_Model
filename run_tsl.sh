#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA13_PATH="/home/alida/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
CUDNN_PATH="/home/alida/.local/lib/python3.12/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDA13_PATH}:${CUDNN_PATH}:${SCRIPT_DIR}/lib/onnxruntime:${LD_LIBRARY_PATH}"
exec "${SCRIPT_DIR}/tsl_translator" "$@"
