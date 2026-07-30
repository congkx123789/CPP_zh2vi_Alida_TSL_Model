#include "translator.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <omp.h>

TSLTranslator::TSLTranslator() : is_ready(false) {}
TSLTranslator::~TSLTranslator() {}

bool TSLTranslator::init(const std::string& base_dir, TSLExecutionMode mode) {
    std::cout << "⚡ Initializing Alida TSL Native C++ Translation Engine..." << std::endl;

    std::string data_dir = base_dir + "/data";
    std::string hv_dict_path = data_dir + "/hanviet.bin";
    {
        std::ifstream test(hv_dict_path);
        if (!test.is_open()) hv_dict_path = data_dir + "/HanViet_CharDict_Enhanced_Merged.txt";
    }
    std::string onnx_model_path = base_dir + "/model/student_nat_int8.onnx";
    {
        std::ifstream test(onnx_model_path);
        if (!test.is_open()) onnx_model_path = base_dir + "/checkpoints/student_nat_int8.onnx";
    }

    // 1. Load Tokenizer
    if (!tokenizer.load(data_dir)) {
        std::cerr << "❌ Failed to load Tokenizer from " << data_dir << std::endl;
        return false;
    }

    // 2. Load Dictionary Trie
    if (!trie.load(data_dir)) {
        std::cerr << "❌ Failed to load Dictionary Trie from " << data_dir << std::endl;
        return false;
    }
    std::cout << "⚡ Engine Dịch Từ Điển: C++ MARISA-Trie DAWG (Flat Store)" << std::endl;

    // 3. Load Logits Processor
    logits_processor = std::make_unique<LogitsProcessor>(tokenizer);
    if (!logits_processor->load_hv_dict(hv_dict_path)) {
        std::cerr << "⚠️ Warning: Failed to load Hán Việt dictionary from " << hv_dict_path << std::endl;
    }

    // 4. Load ONNX Model (CPU, GPU CUDA, or Mobile NPU)
    if (!onnx_engine.load_model(onnx_model_path, mode)) {
        std::cerr << "❌ Failed to load ONNX INT8 model from " << onnx_model_path << std::endl;
        return false;
    }

    is_ready = true;
    warmup();
    return true;
}

void TSLTranslator::warmup() {
    std::cout << "🔥 Running Engine Warmup Routine..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    translate("掌柜在门前等他");
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "✅ Warmup Finished in " << ms << " ms! Engine ready for zero-latency translation." << std::endl;
}

std::string TSLTranslator::translate(const std::string& text_zh) {
    if (!is_ready || text_zh.empty()) return "";

    std::vector<int64_t> zh_ids = tokenizer.encode_zh(text_zh, 64);
    std::vector<MatchInfo> trie_matches = trie.match_sentence(text_zh);

    std::vector<float> logits;
    std::vector<int64_t> logits_shape;
    if (!onnx_engine.run(zh_ids, logits, logits_shape)) {
        return "";
    }

    int seq_len = 64;
    int vocab_size = (logits_shape.size() >= 2) ? logits_shape.back() : 30000;
    if (logits_shape.size() == 3) {
        seq_len = logits_shape[1];
    }

    return logits_processor->process_logits(logits.data(), seq_len, vocab_size, text_zh, trie_matches);
}

std::vector<std::string> TSLTranslator::translate_batch(const std::vector<std::string>& texts_zh, size_t batch_size) {
    double p1, p2, p3;
    return translate_batch_profiled(texts_zh, batch_size, p1, p2, p3);
}

std::vector<std::string> TSLTranslator::translate_batch_profiled(const std::vector<std::string>& texts_zh, size_t batch_size, double& out_p1_ms, double& out_p2_ms, double& out_p3_ms) {
    std::vector<std::string> results(texts_zh.size());
    if (!is_ready || texts_zh.empty()) return results;

    out_p1_ms = 0.0;
    out_p2_ms = 0.0;
    out_p3_ms = 0.0;

    for (size_t b_start = 0; b_start < texts_zh.size(); b_start += batch_size) {
        size_t cur_batch = std::min(batch_size, texts_zh.size() - b_start);

        std::vector<int64_t> batched_ids(cur_batch * 64);
        std::vector<std::vector<MatchInfo>> all_matches(cur_batch);

        // Phase 1: CPU Tokenization & MARISA Trie Matching
        auto tp1_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cur_batch; ++i) {
            const auto& text = texts_zh[b_start + i];
            std::vector<int64_t> ids = tokenizer.encode_zh(text, 64);
            std::copy(ids.begin(), ids.end(), batched_ids.begin() + (i * 64));
            all_matches[i] = trie.match_sentence(text);
        }
        auto tp1_end = std::chrono::high_resolution_clock::now();
        out_p1_ms += std::chrono::duration<double, std::milli>(tp1_end - tp1_start).count();

        // Phase 2: GPU CUDA ONNX Forward Pass
        std::vector<float> batch_logits;
        std::vector<int64_t> logits_shape;
        auto tp2_start = std::chrono::high_resolution_clock::now();
        bool ok = onnx_engine.run_batch(batched_ids, cur_batch, batch_logits, logits_shape);
        auto tp2_end = std::chrono::high_resolution_clock::now();
        out_p2_ms += std::chrono::duration<double, std::milli>(tp2_end - tp2_start).count();

        if (!ok) continue;

        size_t seq_len = 64;
        size_t vocab_size = (logits_shape.size() >= 3) ? logits_shape[2] : 30000;
        size_t stride = seq_len * vocab_size;

        // Phase 3: CPU Candidate Logits Selection & Translation String Formatting
        auto tp3_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cur_batch; ++i) {
            const float* sentence_logits = batch_logits.data() + (i * stride);
            results[b_start + i] = logits_processor->process_logits(
                sentence_logits, seq_len, vocab_size, texts_zh[b_start + i], all_matches[i]
            );
        }
        auto tp3_end = std::chrono::high_resolution_clock::now();
        out_p3_ms += std::chrono::duration<double, std::milli>(tp3_end - tp3_start).count();
    }

    return results;
}
