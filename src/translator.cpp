#include "translator.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <future>
#include <omp.h>

TSLTranslator::TSLTranslator() : exec_mode(TSLExecutionMode::CPU), perf_mode(TSLPerformanceMode::BALANCED_NORMAL), is_ready(false) {}
TSLTranslator::~TSLTranslator() {}

bool TSLTranslator::init(const std::string& base_dir, TSLExecutionMode mode, TSLPerformanceMode performance_mode) {
    exec_mode = mode;
    perf_mode = performance_mode;
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

    // 4. Load ONNX Model (CPU, GPU CUDA, or Mobile NPU with Performance Mode)
    if (!onnx_engine.load_model(onnx_model_path, mode, perf_mode)) {
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

size_t TSLTranslator::get_adaptive_batch_size(size_t total_sentences) const {
    if (total_sentences <= 1) return 1;

    size_t target_batch = 64;

    switch (exec_mode) {
        case TSLExecutionMode::GPU_CUDA:
            if (perf_mode == TSLPerformanceMode::ECO_LOW_POWER) target_batch = 16;
            else if (perf_mode == TSLPerformanceMode::BALANCED_NORMAL) target_batch = 32;
            else target_batch = 32; // Optimized for DirectML 2GB VRAM (NVIDIA MX330 / Intel Iris Xe)
            break;

        case TSLExecutionMode::NPU_COREML:
        case TSLExecutionMode::NPU_QNN:
        case TSLExecutionMode::NPU_NNAPI:
            if (perf_mode == TSLPerformanceMode::ECO_LOW_POWER) target_batch = 8;
            else if (perf_mode == TSLPerformanceMode::BALANCED_NORMAL) target_batch = 16;
            else target_batch = 32;  // MAX_PERFORMANCE
            break;

        case TSLExecutionMode::ARM_XNNPACK:
            if (perf_mode == TSLPerformanceMode::ECO_LOW_POWER) target_batch = 8;
            else if (perf_mode == TSLPerformanceMode::BALANCED_NORMAL) target_batch = 16;
            else target_batch = 32;  // MAX_PERFORMANCE
            break;

        default: // CPU Mode
            if (perf_mode == TSLPerformanceMode::ECO_LOW_POWER) target_batch = 16;
            else if (perf_mode == TSLPerformanceMode::BALANCED_NORMAL) target_batch = 32;
            else target_batch = 64;  // MAX_PERFORMANCE
            break;
    }

    return std::min<size_t>(total_sentences, target_batch);
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
    int vocab_size = 18004;
    if (logits_shape.size() >= 3) {
        seq_len = logits_shape[1];
        vocab_size = logits_shape[2];
    } else if (logits_shape.size() == 2) {
        seq_len = logits_shape[0];
        vocab_size = logits_shape[1];
    }

    return logits_processor->process_logits(logits.data(), seq_len, vocab_size, text_zh, trie_matches);
}

std::vector<std::string> TSLTranslator::translate_batch(const std::vector<std::string>& texts_zh, size_t batch_size) {
    return translate_batch_pipelined(texts_zh, batch_size);
}

struct BatchData {
    size_t start_idx;
    size_t size;
    std::vector<int64_t> batched_ids;
    std::vector<std::vector<MatchInfo>> all_matches;
};

std::vector<std::string> TSLTranslator::translate_batch_pipelined(const std::vector<std::string>& texts_zh, size_t batch_size) {
    std::vector<std::string> results(texts_zh.size());
    if (!is_ready || texts_zh.empty()) return results;

    if (batch_size == 0) {
        batch_size = get_adaptive_batch_size(texts_zh.size());
    }

    size_t total_sentences = texts_zh.size();
    size_t num_batches = (total_sentences + batch_size - 1) / batch_size;

    auto prepare_batch = [&](size_t b_idx) -> BatchData {
        BatchData bd;
        bd.start_idx = b_idx * batch_size;
        bd.size = std::min(batch_size, total_sentences - bd.start_idx);
        bd.batched_ids.resize(bd.size * 64);
        bd.all_matches.resize(bd.size);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < bd.size; ++i) {
            const auto& text = texts_zh[bd.start_idx + i];
            std::vector<int64_t> ids = tokenizer.encode_zh(text, 64);
            std::copy(ids.begin(), ids.end(), bd.batched_ids.begin() + (i * 64));
            bd.all_matches[i] = trie.match_sentence(text);
        }
        return bd;
    };

    // Pre-fetch Batch 0 asynchronously
    std::future<BatchData> next_batch_future = std::async(std::launch::async, prepare_batch, 0);

    for (size_t b = 0; b < num_batches; ++b) {
        // Wait for current batch pre-processing to finish
        BatchData cur_bd = next_batch_future.get();

        // Asynchronously launch pre-fetching for Batch B+1 while GPU processes Batch B
        if (b + 1 < num_batches) {
            next_batch_future = std::async(std::launch::async, prepare_batch, b + 1);
        }

        // GPU Forward Pass
        std::vector<float> batch_logits;
        std::vector<int64_t> logits_shape;
        if (!onnx_engine.run_batch(cur_bd.batched_ids, cur_bd.size, batch_logits, logits_shape)) {
            continue;
        }

        size_t seq_len = 64;
        size_t vocab_size = (logits_shape.size() >= 3) ? logits_shape[2] : 18004;
        size_t stride = seq_len * vocab_size;

        // CPU Logits Selection & Post-processing
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cur_bd.size; ++i) {
            const float* sentence_logits = batch_logits.data() + (i * stride);
            results[cur_bd.start_idx + i] = logits_processor->process_logits(
                sentence_logits, seq_len, vocab_size, texts_zh[cur_bd.start_idx + i], cur_bd.all_matches[i]
            );
        }
    }

    return results;
}

std::vector<std::string> TSLTranslator::translate_batch_profiled(const std::vector<std::string>& texts_zh, size_t batch_size, double& out_p1_ms, double& out_p2_ms, double& out_p3_ms) {
    std::vector<std::string> results(texts_zh.size());
    if (!is_ready || texts_zh.empty()) return results;

    if (batch_size == 0) {
        batch_size = get_adaptive_batch_size(texts_zh.size());
    }

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
        size_t vocab_size = (logits_shape.size() >= 3) ? logits_shape[2] : 18004;
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
