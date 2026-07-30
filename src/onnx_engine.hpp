/**
 * @file onnx_engine.hpp
 * @author Hà Vũ Công
 * @brief Động cơ suy luận C++ ONNX Runtime API cho Mô hình AI Alida TSL INT8 (Hỗ trợ GPU CUDA & CPU)
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
 * @brief Lớp quản lý suy luận ONNX Runtime C API nguyên bản (CPU & GPU CUDA)
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
     * @brief Thực thi suy luận Forward Pass nhận câu đầu vào và trả về ma trận Logits
     * @param input_ids Vector Token IDs câu tiếng Trung (Shape: [1, 64])
     * @param out_logits Vector chứa ma trận xác suất Logits (Shape: [1, 64, 18004])
     * @param out_logits_shape Kích thước ma trận Logits ngõ ra
     * @return true nếu suy luận thành công
     */
    bool run(const std::vector<int64_t>& input_ids, std::vector<float>& out_logits, std::vector<int64_t>& out_logits_shape);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    bool is_loaded;
};

#endif // ONNX_ENGINE_HPP
