#include "onnx_engine.hpp"
#include <iostream>
#include <cstring>
#include "onnxruntime_c_api.h"

struct ONNXInferenceEngine::Impl {
    const OrtApi* g_ort = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* session_options = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memory_info = nullptr;
};

ONNXInferenceEngine::ONNXInferenceEngine() : impl(std::make_unique<Impl>()), is_loaded(false) {}

ONNXInferenceEngine::~ONNXInferenceEngine() {
    if (impl->g_ort) {
        if (impl->memory_info) impl->g_ort->ReleaseMemoryInfo(impl->memory_info);
        if (impl->session) impl->g_ort->ReleaseSession(impl->session);
        if (impl->session_options) impl->g_ort->ReleaseSessionOptions(impl->session_options);
        if (impl->env) impl->g_ort->ReleaseEnv(impl->env);
    }
}

bool ONNXInferenceEngine::load_model(const std::string& model_path, bool use_gpu) {
    impl->g_ort = OrtGetApiBase()->GetApi(17);
    if (!impl->g_ort) {
        std::cerr << "❌ Failed to initialize ONNX Runtime API Base." << std::endl;
        return false;
    }

    OrtStatus* status = impl->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "TSLInference", &impl->env);
    if (status != nullptr) {
        std::cerr << "❌ Failed to create ONNX Env: " << impl->g_ort->GetErrorMessage(status) << std::endl;
        impl->g_ort->ReleaseStatus(status);
        return false;
    }

    status = impl->g_ort->CreateSessionOptions(&impl->session_options);
    if (status != nullptr) {
        impl->g_ort->ReleaseStatus(status);
        return false;
    }
    impl->g_ort->SetIntraOpNumThreads(impl->session_options, 4);
    impl->g_ort->SetSessionGraphOptimizationLevel(impl->session_options, ORT_ENABLE_ALL);

    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        memset(&cuda_options, 0, sizeof(cuda_options));
        cuda_options.device_id = 0;
        OrtStatus* cuda_status = impl->g_ort->SessionOptionsAppendExecutionProvider_CUDA(impl->session_options, &cuda_options);
        if (cuda_status != nullptr) {
            std::cout << "⚠️ Warning: Failed to enable CUDA Provider (" << impl->g_ort->GetErrorMessage(cuda_status) << "). Falling back to CPU Mode." << std::endl;
            impl->g_ort->ReleaseStatus(cuda_status);
        } else {
            std::cout << "🚀 [C++ ONNX Engine] GPU CUDA Execution Provider Active on NVIDIA RTX GPU!" << std::endl;
        }
    } else {
        std::cout << "⚡ [C++ ONNX Engine] CPU Execution Mode Active." << std::endl;
    }

    status = impl->g_ort->CreateSession(impl->env, model_path.c_str(), impl->session_options, &impl->session);
    if (status != nullptr) {
        std::cerr << "❌ Failed to load ONNX Model: " << impl->g_ort->GetErrorMessage(status) << std::endl;
        impl->g_ort->ReleaseStatus(status);
        return false;
    }

    status = impl->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &impl->memory_info);
    if (status != nullptr) {
        impl->g_ort->ReleaseStatus(status);
        return false;
    }

    is_loaded = true;
    return true;
}

bool ONNXInferenceEngine::run(const std::vector<int64_t>& input_ids, std::vector<float>& out_logits, std::vector<int64_t>& out_logits_shape) {
    return run_batch(input_ids, 1, out_logits, out_logits_shape);
}

bool ONNXInferenceEngine::run_batch(const std::vector<int64_t>& batched_ids, size_t batch_size, std::vector<float>& out_logits, std::vector<int64_t>& out_logits_shape) {
    if (!is_loaded || batch_size == 0) return false;

    int64_t input_shape[2] = {static_cast<int64_t>(batch_size), 64};
    OrtValue* input_tensor = nullptr;

    OrtStatus* status = impl->g_ort->CreateTensorWithDataAsOrtValue(
        impl->memory_info,
        const_cast<int64_t*>(batched_ids.data()),
        batched_ids.size() * sizeof(int64_t),
        input_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_tensor
    );

    if (status != nullptr) {
        impl->g_ort->ReleaseStatus(status);
        return false;
    }

    const char* input_names[] = {"src"};
    const char* output_names[] = {"logits", "fertility_pred"};
    OrtValue* output_tensors[2] = {nullptr, nullptr};

    status = impl->g_ort->Run(
        impl->session,
        nullptr,
        input_names,
        (const OrtValue* const*)&input_tensor,
        1,
        output_names,
        2,
        output_tensors
    );

    impl->g_ort->ReleaseValue(input_tensor);

    if (status != nullptr) {
        std::cerr << "❌ ONNX Run Error: " << impl->g_ort->GetErrorMessage(status) << std::endl;
        impl->g_ort->ReleaseStatus(status);
        return false;
    }

    // Get logits output data
    float* logits_ptr = nullptr;
    status = impl->g_ort->GetTensorMutableData(output_tensors[0], (void**)&logits_ptr);
    if (status != nullptr) {
        if (output_tensors[0]) impl->g_ort->ReleaseValue(output_tensors[0]);
        if (output_tensors[1]) impl->g_ort->ReleaseValue(output_tensors[1]);
        impl->g_ort->ReleaseStatus(status);
        return false;
    }

    // Get logits shape
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    status = impl->g_ort->GetTensorTypeAndShape(output_tensors[0], &shape_info);
    if (status == nullptr && shape_info != nullptr) {
        size_t num_dims = 0;
        impl->g_ort->GetDimensionsCount(shape_info, &num_dims);
        out_logits_shape.resize(num_dims);
        impl->g_ort->GetDimensions(shape_info, out_logits_shape.data(), num_dims);
        impl->g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    } else if (status != nullptr) {
        impl->g_ort->ReleaseStatus(status);
    }

    size_t total_elements = 1;
    for (auto d : out_logits_shape) total_elements *= d;

    out_logits.assign(logits_ptr, logits_ptr + total_elements);

    // Release output tensors
    if (output_tensors[0]) impl->g_ort->ReleaseValue(output_tensors[0]);
    if (output_tensors[1]) impl->g_ort->ReleaseValue(output_tensors[1]);

    return true;
}
