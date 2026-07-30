@echo off
chcp 65001 > nul
echo ================================================================================
echo  BUILDING TSL NATIVE C++ TRANSLATOR FOR WINDOWS (LATEST GITHUB CODE)
echo ================================================================================

g++ -O3 -march=native -std=c++17 ^
  -Isrc ^
  -I"lib\marisa\include" ^
  -I"lib\onnxruntime" ^
  -I"lib\onnxruntime\onnxruntime" ^
  src\dictionary.cpp ^
  src\tokenizer.cpp ^
  src\logits_processor.cpp ^
  src\onnx_engine.cpp ^
  src\translator.cpp ^
  src\main.cpp ^
  "lib\marisa\libmarisa.a" ^
  "lib\onnxruntime\libonnxruntime.dll.a" ^
  -o tsl_translator.exe 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ================================================================================
    echo  BUILD FAILED!
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo  [OK] tsl_translator.exe compiled successfully with latest GitHub code!
echo.
echo  Usage:
echo    tsl_translator.exe "中文句子"
echo    tsl_translator.exe --test
echo    tsl_translator.exe --file input.txt --output output.txt
echo    tsl_translator.exe --benchmark
echo ================================================================================
