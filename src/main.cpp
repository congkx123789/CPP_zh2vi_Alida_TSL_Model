#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>
#include "translator.hpp"

int main(int argc, char* argv[]) {
    std::string base_dir = ".";
    std::string input_text = "";
    std::string input_file = "";
    std::string output_file = "";
    bool run_benchmark = false;
    TSLExecutionMode mode = TSLExecutionMode::CPU;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            input_text = argv[++i];
        } else if (std::strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (std::strcmp(argv[i], "--dir") == 0 && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--benchmark") == 0) {
            run_benchmark = true;
        } else if (std::strcmp(argv[i], "--gpu") == 0 || std::strcmp(argv[i], "--cuda") == 0) {
            mode = TSLExecutionMode::GPU_CUDA;
        } else if (std::strcmp(argv[i], "--ane") == 0 || std::strcmp(argv[i], "--coreml") == 0 || std::strcmp(argv[i], "--iphone") == 0) {
            mode = TSLExecutionMode::NPU_COREML;
        } else if (std::strcmp(argv[i], "--qnn") == 0 || std::strcmp(argv[i], "--snapdragon") == 0) {
            mode = TSLExecutionMode::NPU_QNN;
        } else if (std::strcmp(argv[i], "--npu") == 0 || std::strcmp(argv[i], "--nnapi") == 0 || std::strcmp(argv[i], "--android") == 0) {
            mode = TSLExecutionMode::NPU_NNAPI;
        } else if (std::strcmp(argv[i], "--arm") == 0 || std::strcmp(argv[i], "--xnnpack") == 0) {
            mode = TSLExecutionMode::ARM_XNNPACK;
        } else if (std::strcmp(argv[i], "--cpu") == 0) {
            mode = TSLExecutionMode::CPU;
        } else if (input_text.empty() && argv[i][0] != '-') {
            input_text = argv[i];
        }
    }

    TSLTranslator translator;
    if (!translator.init(base_dir, mode)) {
        std::cerr << "❌ Failed to initialize TSL Native C++ Translator." << std::endl;
        return 1;
    }

    if (run_benchmark) {
        std::string mode_name = "CPU MODE";
        if (mode == TSLExecutionMode::GPU_CUDA) mode_name = "NVIDIA GPU CUDA TENSOR CORES";
        else if (mode == TSLExecutionMode::NPU_COREML) mode_name = "APPLE NEURAL ENGINE (ANE CoreML)";
        else if (mode == TSLExecutionMode::NPU_QNN) mode_name = "QUALCOMM SNAPDRAGON HEXAGON NPU (QNN SDK)";
        else if (mode == TSLExecutionMode::NPU_NNAPI) mode_name = "ANDROID UNIVERSAL NPU (NNAPI)";
        else if (mode == TSLExecutionMode::ARM_XNNPACK) mode_name = "ARM MOBILE LOW-POWER ENGINE (XNNPACK)";

        std::cout << "\n================================================================================" << std::endl;
        std::cout << "🚀 PHÂN TÍCH PHÂN CẤP ĐIỂM NGHẼN (PROFILING 1,000 CÂU): " << mode_name << std::endl;
        std::cout << "================================================================================" << std::endl;

        std::vector<std::string> base_sentences = {
            "掌柜在门前等他",
            "一言既出，驷马难追",
            "三三两两的人群",
            "5000两白银",
            "一头扎进九层楼",
            "李云飞大怒道：“掌柜在门前等 hắn，一言既出，驷马难追！”",
            "第三百五十六章 5000两白银!",
            " hắn 一头扎进九层楼，向前方走去。",
            "修仙者的路是极其艰难的，他经历了无数的磨难。",
            "他拿起了宗门的飞剑，快步地走了。"
        };

        // Fast Benchmark: 1,000 sentences total
        int total_runs = 1000;
        std::vector<std::string> test_sentences(total_runs);
        for (int r = 0; r < total_runs; ++r) {
            test_sentences[r] = base_sentences[r % base_sentences.size()];
        }

        // Sequential Single Sentence Test (100 sentences)
        int seq_runs = 100;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < seq_runs; ++r) {
            translator.translate(base_sentences[r % base_sentences.size()]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double seq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double seq_tps = (seq_runs / seq_ms) * 1000.0;
        double seq_latency = seq_ms / seq_runs;

        std::cout << "⚡ [Sequential Mode] 100 câu  | Thời gian: " << std::fixed << std::setprecision(3) << (seq_ms / 1000.0)
                  << "s | Độ trễ: " << seq_latency << " ms/câu | Băng thông: " << std::setprecision(1) << seq_tps << " câu/giây\n" << std::endl;

        // Bottleneck Profiling across Batch Sizes (32, 64, 128)
        std::vector<size_t> batch_sizes = {32, 64, 128};

        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "🔍 PHÂN TÍCH CHI TIẾT THỜI GIAN TIÊU TỐN TỪNG TRẠM (1,000 CÂU)" << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;

        for (size_t bs : batch_sizes) {
            double p1_ms = 0.0, p2_ms = 0.0, p3_ms = 0.0;
            auto tb0 = std::chrono::high_resolution_clock::now();
            auto results = translator.translate_batch_profiled(test_sentences, bs, p1_ms, p2_ms, p3_ms);
            auto tb1 = std::chrono::high_resolution_clock::now();

            double total_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
            double b_tps = (total_runs / total_ms) * 1000.0;

            double p1_pct = (p1_ms / total_ms) * 100.0;
            double p2_pct = (p2_ms / total_ms) * 100.0;
            double p3_pct = (p3_ms / total_ms) * 100.0;

            std::cout << "📊 BATCH " << std::setw(3) << bs << " (Tổng thời gian: " << std::setprecision(3) << (total_ms / 1000.0) << "s | Băng thông: " << std::setprecision(1) << b_tps << " câu/s)" << std::endl;
            std::cout << "   ├─ Trạm 1 (CPU Tokenizer + Trie Match) : " << std::setprecision(3) << (p1_ms / 1000.0) << "s (" << std::setprecision(1) << p1_pct << "%)" << std::endl;
            std::cout << "   ├─ Trạm 2 (GPU CUDA ONNX Tensor Run)  : " << std::setprecision(3) << (p2_ms / 1000.0) << "s (" << std::setprecision(1) << p2_pct << "%) [TẢI GPU THẬT]" << std::endl;
            std::cout << "   └─ Trạm 3 (CPU Logits & String Build) : " << std::setprecision(3) << (p3_ms / 1000.0) << "s (" << std::setprecision(1) << p3_pct << "%)\n" << std::endl;
        }

        std::cout << "================================================================================" << std::endl;
        return 0;
    }

    if (!input_file.empty()) {
        std::ifstream f_in(input_file);
        if (!f_in.is_open()) {
            std::cerr << "❌ Failed to open input file: " << input_file << std::endl;
            return 1;
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(f_in, line)) {
            lines.push_back(line);
        }
        f_in.close();

        std::cout << "📁 Total lines read from " << input_file << ": " << lines.size() << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::string> translated_lines = translator.translate_batch(lines, 256);

        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!output_file.empty()) {
            std::ofstream f_out(output_file);
            for (const auto& l : translated_lines) {
                f_out << l << "\n";
            }
            f_out.close();
            std::cout << "📦 Saved translated file: " << output_file << std::endl;
        } else {
            for (size_t i = 0; i < std::min<size_t>(10, translated_lines.size()); ++i) {
                std::cout << translated_lines[i] << "\n";
            }
        }

        std::cout << "✅ Finished translating " << lines.size() << " lines in " << (total_ms / 1000.0) << "s (" << (lines.size() / (total_ms / 1000.0)) << " lines/sec)" << std::endl;
        return 0;
    }

    if (input_text.empty()) {
        input_text = "李云飞大怒道：“掌柜在门前等他，一言既出，驷马难追！”";
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "🇨🇳 GỐC : " << input_text << std::endl;
    std::string translated = translator.translate(input_text);
    std::cout << "🇻🇳 DỊCH: " << translated << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
