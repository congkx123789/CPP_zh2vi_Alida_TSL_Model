#ifndef TRANSLATOR_HPP
#define TRANSLATOR_HPP

#include <string>
#include <memory>
#include "tokenizer.hpp"
#include "dictionary.hpp"
#include "onnx_engine.hpp"
#include "logits_processor.hpp"

class TSLTranslator {
public:
    TSLTranslator();
    ~TSLTranslator();

    bool init(const std::string& base_dir);
    std::string translate(const std::string& text_zh);

private:
    TranslationTokenizer tokenizer;
    VietphraseTrie trie;
    ONNXInferenceEngine onnx_engine;
    std::unique_ptr<LogitsProcessor> logits_processor;
    bool is_ready;

    void warmup();
};

#endif // TRANSLATOR_HPP
