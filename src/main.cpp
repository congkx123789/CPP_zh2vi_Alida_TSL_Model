#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include "translator.hpp"

int main(int argc, char* argv[]) {
    std::string base_dir = ".";
    std::string input_text = "";
    std::string input_file = "";
    std::string output_file = "";
    bool run_benchmark = false;
    bool use_gpu = false;

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
        } else if (std::strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (std::strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
        } else if (input_text.empty() && argv[i][0] != '-') {
            input_text = argv[i];
        }
    }

    TSLTranslator translator;
    if (!translator.init(base_dir, use_gpu)) {
        std::cerr << "❌ Failed to initialize TSL Native C++ Translator." << std::endl;
        return 1;
    }

    if (run_benchmark) {
        std::cout << "\n================================================================================" << std::endl;
        std::cout << "🚀 CHƯƠNG TRÌNH BENCHMARK TỐC ĐỘ NGUYÊN BẢN C++ (" << (use_gpu ? "GPU CUDA BATCHING" : "CPU MODE") << ")" << std::endl;
        std::cout << "================================================================================" << std::endl;

        std::vector<std::string> base_sentences = {
            "掌柜在门前等他",
            "一言既出，驷马难追",
            "三三两两的人群",
            "5000两白银",
            "一头扎进九层楼",
            "李云飞大怒道：“掌柜在门前等 hắn，一言既出，驷马难追！”",
            "第三百五十六章 5000两白银!",
            " hắn 一头扎进九层楼，向前方走去。"
        };

        int total_runs = 5000;
        std::vector<std::string> test_sentences(total_runs);
        for (int r = 0; r < total_runs; ++r) {
            test_sentences[r] = base_sentences[r % base_sentences.size()];
        }

        // Test 1: Single Sentence Sequential Mode
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < 500; ++r) {
            translator.translate(test_sentences[r]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double seq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double seq_tps = (500.0 / seq_ms) * 1000.0;

        std::cout << "⚡ [C++ Sequential Mode] 500 câu              : " << (seq_ms / 1000.0) << "s (" << seq_tps << " câu/giây)" << std::endl;

        // Test 2: Parallel Batch GPU Mode
        auto t2 = std::chrono::high_resolution_clock::now();
        auto batch_results = translator.translate_batch(test_sentences, 256);
        auto t3 = std::chrono::high_resolution_clock::now();

        double batch_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double batch_tps = (total_runs / batch_ms) * 1000.0;

        std::cout << "🔥 [C++ Parallel GPU Batch 256] " << total_runs << " câu      : " << (batch_ms / 1000.0) << "s (" << batch_tps << " câu/giây)" << std::endl;
        std::cout << "🚀 Tăng tốc vượt trội                     : " << (batch_tps / seq_tps) << "x lần so với dịch tuần tự" << std::endl;
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
