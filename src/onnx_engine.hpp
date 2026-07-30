/**
 * @file onnx_engine.hpp
 * @author Hà Vũ Công
 * @brief Động cơ suy luận C++ ONNX Runtime API hỗ trợ đầy đủ GPU CUDA và NPU Điện Thoại Di Động
 * (Apple Neural Engine ANE, Qualcomm Hexagon NPU, MediaTek APU, Samsung NPU & Google Tensor TPU)
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
 * @brief Chế độ phần cứng tăng tốc suy luận trên Máy tính & Điện thoại di động
 */
enum class TSLExecutionMode {
    CPU,            ///< Chế độ CPU chuẩn (x86/x64)
    GPU_CUDA,       ///< Chế độ GPU NVIDIA CUDA (Tensor Cores)
    NPU_NNAPI,      ///< Android Universal NPU (MediaTek APU, Samsung NPU, Google Tensor TPU)
    NPU_QNN,        ///< Qualcomm Snapdragon Hexagon NPU (QNN SDK)
    NPU_COREML,     ///< Apple Neural Engine (ANE) trên iPhone / iPad (CoreML)
    ARM_XNNPACK     ///< ARM Mobile High-Efficiency Neon Engine (Tối ưu hóa Pin)
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
     * @brief Nạp mô hình INT8 ONNX từ đĩa (Hỗ trợ tăng tốc GPU CUDA & NPU Điện thoại)
     * @param model_path Đường dẫn tới file student_nat_int8.onnx
     * @param mode Chế độ phần cứng thực thi (CPU / GPU_CUDA / NPU_NNAPI / NPU_QNN / NPU_COREML / ARM_XNNPACK)
     * @return true nếu nạp thành công
     */
    bool load_model(const std::string& model_path, TSLExecutionMode mode = TSLExecutionMode::CPU);

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
