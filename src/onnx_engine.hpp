/**
 * @file onnx_engine.hpp
 * @author Hà Vũ Công
 * @brief Động cơ suy luận C++ ONNX Runtime API cho Mô hình AI Alida TSL INT8 (Hỗ trợ GPU CUDA & CPU Batch)
 * @url https://github.com/congkx123789/CPP_zh2vi_Alida_TSL_Model
 */

#ifndef ONNX_ENGINE_HPP
#define ONNX_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

/**
 * @class ONNXInferenceEngine
 * @brief Lớp quản lý suy luận ONNX Runtime C API nguyên bản (CPU & GPU CUDA Batching)
 */
class ONNXInferenceEngine {
public:
    ONNXInferenceEngine();
    ~ONNXInferenceEngine();

    /**
     * @brief Nạp mô hình INT8 ONNX từ đĩa (Hỗ trợ tăng tốc GPU CUDA)
     * @param model_path Đường dẫn tới file student_nat_int8.onnx
     * @param use_gpu Đặt true để dùng GPU CUDA Execution Provider
     * @return true nếu nạp thành công
     */
    bool load_model(const std::string& model_path, bool use_gpu = false);

    /**
     * @brief Thực thi suy luận Forward Pass cho 1 câu (Shape: [1, 64])
     */
    bool run(const std::vector<int64_t>& input_ids, std::vector<float>& out_logits, std::vector<int64_t>& out_logits_shape);

    /**
     * @brief Thực thi suy luận Forward Pass song song cho Batch N câu trên GPU CUDA (Shape: [Batch_Size, 64])
     */
    bool run_batch(const std::vector<int64_t>& batched_ids, size_t batch_size, std::vector<float>& out_logits, std::vector<int64_t>& out_logits_shape);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    bool is_loaded;
};

#endif // ONNX_ENGINE_HPP
