#!/bin/bash
set -e

echo "================================================================================"
echo "🔨 BUILDING TSL NATIVE C++ TRANSLATOR"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Compile
g++ -O3 -std=c++17 \
  -Isrc \
  -Ilib/marisa/include \
  -Ilib/onnxruntime/onnxruntime \
  src/dictionary.cpp \
  src/tokenizer.cpp \
  src/logits_processor.cpp \
  src/onnx_engine.cpp \
  src/translator.cpp \
  src/main.cpp \
  lib/marisa/libmarisa.a \
  lib/onnxruntime/libonnxruntime.so.1 \
  -Wl,-rpath,'$ORIGIN/lib/onnxruntime' \
  -Wl,-rpath,/home/alida/.local/lib/python3.12/site-packages/onnxruntime/capi \
  -o tsl_translator

echo "================================================================================"
echo "✅ BUILD SUCCESSFUL! Binary: ./tsl_translator"
echo ""
echo "Usage:"
echo "  ./tsl_translator \"中文句子\"              # Dịch 1 câu"
echo "  ./tsl_translator --text \"中文句子\"        # Dịch 1 câu"
echo "  ./tsl_translator --file input.txt         # Dịch file"
echo "  ./tsl_translator --benchmark              # Benchmark tốc độ"
echo "================================================================================"
