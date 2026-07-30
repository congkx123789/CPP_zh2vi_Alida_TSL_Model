/**
 * @file onnx_engine.hpp
 * @author Hà Vũ Công
 * @brief Động cơ suy luận C++ ONNX Runtime API hỗ trợ đầy đủ GPU CUDA và NPU Điện Thoại Di Động
 * @url https://github.com/congkx123789/CPP_zh2vi_Alida_TSL_Model
 */

#ifndef ONNX_ENGINE_HPP
#define ONNX_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

/**
 * @enum TSLExecutionMode
 * @brief Chế độ phần cứng thực thi
 */
enum class TSLExecutionMode {
    CPU,            ///< Chế độ CPU chuẩn
    GPU_CUDA,       ///< Chế độ GPU NVIDIA CUDA (Tensor Cores sm_120 Target)
    NPU_NNAPI,      ///< Android Universal NPU (MediaTek APU, Samsung NPU, Google Tensor TPU)
    NPU_QNN,        ///< Qualcomm Snapdragon Hexagon NPU (QNN SDK)
    NPU_COREML,     ///< Apple Neural Engine (ANE) trên iPhone / iPad (CoreML)
    ARM_XNNPACK     ///< ARM Mobile High-Efficiency Neon Engine
};

/**
 * @enum TSLPerformanceMode
 * @brief Chế độ cấu hình hiệu năng & tiêu thụ điện năng
 */
enum class TSLPerformanceMode {
    ECO_LOW_POWER,   ///< Chế độ Tiết kiệm Pin / Tải thấp (2 threads, Batch 16)
    BALANCED_NORMAL, ///< Chế độ Cân bằng Thông thường (4 threads, Batch 64)
    MAX_PERFORMANCE  ///< Chế độ Hiệu năng Cực hạn (Tối đa Cores CPU/GPU, Batch 128)
};

/**
 * @class ONNXInferenceEngine
 * @brief Lớp quản lý suy luận ONNX Runtime C API nguyên bản (CPU, GPU CUDA & Mobile NPU)
 */
class ONNXInferenceEngine {
public:
    ONNXInferenceEngine();
    ~ONNXInferenceEngine();

    /**
     * @brief Nạp mô hình INT8 ONNX từ đĩa với cấu hình Chế độ Hiệu năng
     * @param model_path Đường dẫn tới file student_nat_int8.onnx
     * @param mode Chế độ phần cứng thực thi
     * @param perf_mode Chế độ hiệu năng (ECO / BALANCED / MAX_PERFORMANCE)
     * @return true nếu nạp thành công
     */
    bool load_model(const std::string& model_path, TSLExecutionMode mode = TSLExecutionMode::CPU, TSLPerformanceMode perf_mode = TSLPerformanceMode::BALANCED_NORMAL, int device_id = 0);

    /**
     * @brief Thực thi suy luận Forward Pass cho 1 câu (Shape: [1, 64])
     */
    bool run(const std::vector<int64_t>& input_ids, std::vector<float>& out_logits, std::vector<int64_t>& out_logits_shape);

    /**
     * @brief Thực thi suy luận Forward Pass song song cho Batch N câu trên GPU CUDA / Mobile NPU (Shape: [Batch_Size, 64])
     */
    bool run_batch(const std::vector<int64_t>& batched_ids, size_t batch_size, std::vector<float>& out_logits, std::vector<int64_t>& out_logits_shape);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    bool is_loaded;
};

#endif // ONNX_ENGINE_HPP
