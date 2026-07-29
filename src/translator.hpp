/**
 * @file translator.hpp
 * @author Hà Vũ Công
 * @brief Bộ điều phối tổng thể Động cơ Dịch thuật Alida TSL C++ 3 Trạm
 * @url https://github.com/congkx123789/CPP_zh2vi_Alida_TSL_Model
 */

#ifndef TRANSLATOR_HPP
#define TRANSLATOR_HPP

#include <string>
#include <memory>
#include "tokenizer.hpp"
#include "dictionary.hpp"
#include "onnx_engine.hpp"
#include "logits_processor.hpp"

/**
 * @class TSLTranslator
 * @brief Lớp dịch thuật cấp cao tích hợp toàn bộ Pipeline 3 Trạm C++ Nguyên bản
 */
class TSLTranslator {
public:
    TSLTranslator();
    ~TSLTranslator();

    /**
     * @brief Khởi tạo toàn bộ động cơ dịch (nạp Tokenizer, DAWG Trie, ONNX Model, Hán Việt Dict)
     * @param base_dir Thư mục gốc chứa data/ và model/
     * @return true nếu khởi tạo thành công
     */
    bool init(const std::string& base_dir = ".");

    /**
     * @brief Dịch một câu tiếng Trung sang tiếng Việt hoàn chỉnh
     * @param text_zh Câu tiếng Trung nguyên bản
     * @return Chuỗi tiếng Việt đã qua xử lý Trạm 1, 2 & 3
     */
    std::string translate(const std::string& text_zh);

private:
    void warmup();

    TranslationTokenizer tokenizer;
    VietphraseTrie trie;
    ONNXInferenceEngine onnx_engine;
    std::unique_ptr<LogitsProcessor> logits_processor;
    bool is_ready;
};

#endif // TRANSLATOR_HPP
