/**
 * @file translator.hpp
 * @author Hà Vũ Công
 * @brief Bộ điều phối tổng thể Động cơ Dịch thuật Alida TSL C++ 3 Trạm
 * @url https://github.com/congkx123789/CPP_zh2vi_Alida_TSL_Model
 */

#ifndef TRANSLATOR_HPP
#define TRANSLATOR_HPP

#include <string>
#include <vector>
#include <memory>
#include "tokenizer.hpp"
#include "dictionary.hpp"
#include "onnx_engine.hpp"
#include "logits_processor.hpp"

/**
 * @class TSLTranslator
 * @brief Lớp dịch thuật cấp cao tích hợp toàn bộ Pipeline 3 Trạm C++ Nguyên bản (Hỗ trợ CPU, GPU CUDA & Mobile NPU)
 */
class TSLTranslator {
public:
    TSLTranslator();
    ~TSLTranslator();

    /**
     * @brief Khởi tạo toàn bộ động cơ dịch (nạp Tokenizer, DAWG Trie, ONNX Model, Hán Việt Dict)
     * @param base_dir Thư mục gốc chứa data/ và model/
     * @param mode Chế độ phần cứng thực thi
     * @param perf_mode Chế độ hiệu năng (ECO / BALANCED / MAX_PERFORMANCE)
     * @return true nếu khởi tạo thành công
     */
    bool init(const std::string& base_dir = ".", TSLExecutionMode mode = TSLExecutionMode::CPU, TSLPerformanceMode perf_mode = TSLPerformanceMode::BALANCED_NORMAL, int device_id = 0);

    /**
     * @brief Dịch một câu tiếng Trung sang tiếng Việt hoàn chỉnh
     * @param text_zh Câu tiếng Trung nguyên bản
     * @return Chuỗi tiếng Việt đã qua xử lý Trạm 1, 2 & 3
     */
    std::string translate(const std::string& text_zh);

    /**
     * @brief Dịch song song một mảng danh sách các câu với Batch Size tự động thích ứng phần cứng
     * @param texts_zh Danh sách các câu tiếng Trung
     * @param batch_size Kích thước Batch (mặc định: 0 = Tự động tối ưu theo GPU/NPU và Perf Mode)
     * @return Mảng danh sách các câu tiếng Việt tương ứng
     */
    std::vector<std::string> translate_batch(const std::vector<std::string>& texts_zh, size_t batch_size = 0);

    /**
     * @brief Dịch theo cơ chế Pipeline Đôi (Double-Buffering Async Pipelining)
     */
    std::vector<std::string> translate_batch_pipelined(const std::vector<std::string>& texts_zh, size_t batch_size = 0);

    /**
     * @brief Đo chi tiết từng Trạm (Bottleneck Profiling) để tìm chính xác điểm nghẽn hiệu năng
     */
    std::vector<std::string> translate_batch_profiled(const std::vector<std::string>& texts_zh, size_t batch_size, double& out_p1_ms, double& out_p2_ms, double& out_p3_ms);

    /**
     * @brief Tính toán Batch Size tối ưu tự động dựa theo loại phần cứng và chế độ hiệu năng
     */
    size_t get_adaptive_batch_size(size_t total_sentences) const;

private:
    void warmup();

    TranslationTokenizer tokenizer;
    VietphraseTrie trie;
    ONNXInferenceEngine onnx_engine;
    std::unique_ptr<LogitsProcessor> logits_processor;
    TSLExecutionMode exec_mode;
    TSLPerformanceMode perf_mode;
    bool is_ready;
};

#endif // TRANSLATOR_HPP
