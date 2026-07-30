#!/bin/bash
set -e

echo "================================================================================"
echo "🔨 BUILDING TSL NATIVE C++ TRANSLATOR (CPU & GPU CUDA SUPPORT + OPENMP)"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CUDA13_PATH="/home/alida/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
CUBLAS_PATH="/home/alida/.local/lib/python3.12/site-packages/nvidia/cublas/lib"
CUDNN_PATH="/home/alida/.local/lib/python3.12/site-packages/nvidia/cudnn/lib"
ORT_CAPI_PATH="$SCRIPT_DIR/lib/onnxruntime"

# Compile with OpenMP & GPU CUDA ONNX Runtime support
g++ -O3 -std=c++17 -fopenmp \
  -Isrc \
  -Ilib/marisa/include \
  -Ilib/onnxruntime/onnxruntime \
  -Ilib/onnxruntime \
  src/dictionary.cpp \
  src/tokenizer.cpp \
  src/logits_processor.cpp \
  src/onnx_engine.cpp \
  src/translator.cpp \
  src/main.cpp \
  lib/marisa/libmarisa.a \
  "${ORT_CAPI_PATH}/libonnxruntime.so.1.27.0" \
  -Wl,-rpath,'$ORIGIN/lib/onnxruntime' \
  -Wl,-rpath,"${ORT_CAPI_PATH}" \
  -Wl,-rpath,"${CUDA13_PATH}" \
  -Wl,-rpath,"${CUBLAS_PATH}" \
  -Wl,-rpath,"${CUDNN_PATH}" \
  -o tsl_translator

# Create wrapper script for automatic CUDA library path resolution
cat << 'EOF' > run_tsl.sh
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA13_PATH="/home/alida/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
CUDNN_PATH="/home/alida/.local/lib/python3.12/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDA13_PATH}:${CUDNN_PATH}:${SCRIPT_DIR}/lib/onnxruntime:${LD_LIBRARY_PATH}"
exec "${SCRIPT_DIR}/tsl_translator" "$@"
EOF
chmod +x run_tsl.sh

echo "================================================================================"
echo "✅ BUILD SUCCESSFUL! Binary: ./tsl_translator & ./run_tsl.sh"
echo ""
echo "Usage:"
echo "  ./run_tsl.sh \"中文句子\"              # Dịch 1 câu (CPU Mode)"
echo "  ./run_tsl.sh --gpu \"中文句子\"        # Dịch 1 câu (GPU CUDA Mode)"
echo "  ./run_tsl.sh --file input.txt         # Dịch file"
echo "  ./run_tsl.sh --benchmark --gpu        # Benchmark GPU CUDA"
echo "================================================================================"
