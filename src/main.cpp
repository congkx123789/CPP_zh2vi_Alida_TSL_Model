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
        std::cout << "🚀 CHƯƠNG TRÌNH BENCHMARK TỐC ĐỘ NGUYÊN BẢN C++ (" << (use_gpu ? "GPU CUDA" : "CPU") << ")" << std::endl;
        std::cout << "================================================================================" << std::endl;

        std::vector<std::string> test_sentences = {
            "掌柜在门前等他",
            "一言既出，驷马难追",
            "三三两两的人群",
            "5000两白银",
            "一头扎进九层楼",
            "李云飞大怒道：“掌柜在门前等 hắn，一言既出，驷马难追！”",
            "第三百五十六章 5000两白银!",
            " hắn 一头扎进九层楼，向前方走去。"
        };

        int total_runs = 1000;
        int num_sentences = test_sentences.size();
        
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < total_runs; ++r) {
            const auto& s = test_sentences[r % num_sentences];
            translator.translate(s);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avg_ms = total_ms / total_runs;
        double throughput = (total_runs / total_ms) * 1000.0;

        std::cout << "⚡ Tổng số câu test                 : " << total_runs << " câu" << std::endl;
        std::cout << "⚡ Tổng thời gian thực thi         : " << (total_ms / 1000.0) << " giây" << std::endl;
        std::cout << "⚡ Độ trễ trung bình mỗi câu       : " << avg_ms << " ms / câu" << std::endl;
        std::cout << "🔥 BĂNG THÔNG DỊCH C++ (THROUGHPUT): " << throughput << " câu / giây (~" << (throughput * 60.0) << " câu/phút)" << std::endl;
        std::cout << "================================================================================" << std::endl;
        return 0;
    }

    if (!input_file.empty()) {
        std::ifstream f_in(input_file);
        if (!f_in.is_open()) {
            std::cerr << "❌ Failed to open input file: " << input_file << std::endl;
            return 1;
        }

        std::ofstream f_out;
        if (!output_file.empty()) {
            f_out.open(output_file);
        }

        std::string line;
        int count = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        while (std::getline(f_in, line)) {
            if (line.empty()) {
                if (f_out.is_open()) f_out << "\n";
                else std::cout << "\n";
                continue;
            }
            std::string res = translator.translate(line);
            if (f_out.is_open()) {
                f_out << res << "\n";
            } else {
                std::cout << res << "\n";
            }
            count++;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "✅ Finished translating " << count << " lines in " << (total_ms / 1000.0) << "s (" << (total_ms / count) << " ms/line)" << std::endl;
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
