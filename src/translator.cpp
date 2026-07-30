#include "translator.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

TSLTranslator::TSLTranslator() : is_ready(false) {}
TSLTranslator::~TSLTranslator() {}

bool TSLTranslator::init(const std::string& base_dir, bool use_gpu) {
    std::cout << "⚡ Initializing Alida TSL Native C++ Translation Engine..." << std::endl;

    std::string data_dir = base_dir + "/data";
    std::string hv_dict_path = data_dir + "/hanviet.bin";
    {
        std::ifstream test(hv_dict_path);
        if (!test.is_open()) hv_dict_path = data_dir + "/HanViet_CharDict_Enhanced_Merged.txt";
    }
    // Try model/ first, then checkpoints/
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

    // 4. Load ONNX Model (CPU or GPU)
    if (!onnx_engine.load_model(onnx_model_path, use_gpu)) {
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

    // Step 1: Tokenizer Encode & Trie Match
    std::vector<int64_t> zh_ids = tokenizer.encode_zh(text_zh, 64);
    std::vector<MatchInfo> trie_matches = trie.match_sentence(text_zh);

    // Step 2: ONNX Model Forward Pass
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

    // Step 3: Logits Processor & Output Generation
    return logits_processor->process_logits(logits.data(), seq_len, vocab_size, text_zh, trie_matches);
}
