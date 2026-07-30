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
    bool run_test_suite = false;
    TSLExecutionMode mode = TSLExecutionMode::CPU;
    TSLPerformanceMode perf_mode = TSLPerformanceMode::MAX_PERFORMANCE;

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
        } else if (std::strcmp(argv[i], "--test") == 0 || std::strcmp(argv[i], "--suite") == 0) {
            run_test_suite = true;
        } else if (std::strcmp(argv[i], "--gpu") == 0 || std::strcmp(argv[i], "--cuda") == 0) {
            mode = TSLExecutionMode::GPU_CUDA;
            if (i + 1 < argc && argv[i + 1][0] != '-') input_text = argv[++i];
        } else if (std::strcmp(argv[i], "--ane") == 0 || std::strcmp(argv[i], "--coreml") == 0 || std::strcmp(argv[i], "--iphone") == 0) {
            mode = TSLExecutionMode::NPU_COREML;
            if (i + 1 < argc && argv[i + 1][0] != '-') input_text = argv[++i];
        } else if (std::strcmp(argv[i], "--qnn") == 0 || std::strcmp(argv[i], "--snapdragon") == 0) {
            mode = TSLExecutionMode::NPU_QNN;
            if (i + 1 < argc && argv[i + 1][0] != '-') input_text = argv[++i];
        } else if (std::strcmp(argv[i], "--npu") == 0 || std::strcmp(argv[i], "--nnapi") == 0 || std::strcmp(argv[i], "--android") == 0) {
            mode = TSLExecutionMode::NPU_NNAPI;
            if (i + 1 < argc && argv[i + 1][0] != '-') input_text = argv[++i];
        } else if (std::strcmp(argv[i], "--arm") == 0 || std::strcmp(argv[i], "--xnnpack") == 0) {
            mode = TSLExecutionMode::ARM_XNNPACK;
            if (i + 1 < argc && argv[i + 1][0] != '-') input_text = argv[++i];
        } else if (std::strcmp(argv[i], "--cpu") == 0) {
            mode = TSLExecutionMode::CPU;
            if (i + 1 < argc && argv[i + 1][0] != '-') input_text = argv[++i];
        } else if (std::strcmp(argv[i], "--eco") == 0 || std::strcmp(argv[i], "--low-power") == 0) {
            perf_mode = TSLPerformanceMode::ECO_LOW_POWER;
        } else if (std::strcmp(argv[i], "--normal") == 0 || std::strcmp(argv[i], "--balanced") == 0) {
            perf_mode = TSLPerformanceMode::BALANCED_NORMAL;
        } else if (std::strcmp(argv[i], "--max") == 0 || std::strcmp(argv[i], "--performance") == 0) {
            perf_mode = TSLPerformanceMode::MAX_PERFORMANCE;
        } else if (input_text.empty() && argv[i][0] != '-') {
            input_text = argv[i];
        }
    }

    TSLTranslator translator;
    if (!translator.init(base_dir, mode, perf_mode)) {
        std::cerr << "❌ Failed to initialize TSL Native C++ Translator." << std::endl;
        return 1;
    }

    std::vector<std::string> test_suite_sentences = {
        "掌柜在门前等他",
        "一言既出，驷马难追",
        "三三两两的人群",
        "5000两白银",
        "一头扎进九层楼",
        "李云飞大怒道：“掌柜在门前等他，一言既出，驷马难追！”",
        "第三百五十六章 5000两白银!",
        "他一头扎进九层楼，向前方走去。",
        "修仙者的路是极其艰难的，他经历了无数的磨难。",
        "他拿起了宗门的飞剑，快步地走了。",
        "叶凡盘膝而坐，运转体内极其雄厚的元力。",
        "这株千年灵药乃是不可多得的稀世珍宝。",
        "长老高声道：“今日乃是宗门大比之日！”",
        "虚空中无尽的雷劫疯狂地下落。",
        "经过漫长的修炼，他终于成功突破到了金丹境界。",
        "他服下了一枚九品洗髓丹。",
        "城门外聚集了成千上万的修仙者。",
        "林枫淡然一笑：“就凭你这三脚猫的功夫？”",
        "一道刺眼的剑光划破了长空。",
        "天地万物皆在这一刻静止了。"
    };

    if (run_test_suite) {
        std::cout << "\n================================================================================" << std::endl;
        std::cout << "🧪 CHẠY BỘ TEST SUITE 20 CÂU TIỂU THUYẾT TIÊN HIỆP ĐA DẠNG (GPU / NPU / CPU)" << std::endl;
        std::cout << "================================================================================" << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::string> results = translator.translate_batch(test_suite_sentences, 0);
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        for (size_t i = 0; i < test_suite_sentences.size(); ++i) {
            std::cout << "[" << std::setw(2) << (i + 1) << "] 🇨🇳 GỐC : " << test_suite_sentences[i] << std::endl;
            std::cout << "     🇻🇳 DỊCH: " << results[i] << "\n" << std::endl;
        }

        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "✅ Đã dịch thành công 20 câu trong " << std::setprecision(3) << (total_ms / 1000.0) << "s ("
                  << std::setprecision(1) << (total_ms / test_suite_sentences.size()) << " ms/câu)" << std::endl;
        std::cout << "================================================================================\n" << std::endl;
        return 0;
    }

    if (run_benchmark) {
        std::string mode_name = "CPU MODE";
        if (mode == TSLExecutionMode::GPU_CUDA) mode_name = "NVIDIA GPU CUDA TENSOR CORES (sm_120 Target)";
        else if (mode == TSLExecutionMode::NPU_COREML) mode_name = "APPLE NEURAL ENGINE (ANE CoreML)";
        else if (mode == TSLExecutionMode::NPU_QNN) mode_name = "QUALCOMM SNAPDRAGON HEXAGON NPU (QNN SDK)";
        else if (mode == TSLExecutionMode::NPU_NNAPI) mode_name = "ANDROID UNIVERSAL NPU (NNAPI)";
        else if (mode == TSLExecutionMode::ARM_XNNPACK) mode_name = "ARM MOBILE LOW-POWER ENGINE (XNNPACK)";

        std::cout << "\n================================================================================" << std::endl;
        std::cout << "🚀 PHÂN TÍCH CHUYÊN SÂU TẢI PHẦN CỨNG 3 CHẾ ĐỘ HIỆU NĂNG (ECO / NORMAL / MAX)" << std::endl;
        std::cout << "================================================================================" << std::endl;

        int total_runs = 1000;
        std::vector<std::string> test_sentences(total_runs);
        for (int r = 0; r < total_runs; ++r) {
            test_sentences[r] = test_suite_sentences[r % test_suite_sentences.size()];
        }

        // Sequential Single Sentence Test
        int seq_runs = 100;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < seq_runs; ++r) {
            translator.translate(test_suite_sentences[r % test_suite_sentences.size()]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double seq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double seq_tps = (seq_runs / seq_ms) * 1000.0;
        double seq_latency = seq_ms / seq_runs;

        std::cout << "⚡ [Sequential Mode] 100 câu | Thời gian: " << std::fixed << std::setprecision(3) << (seq_ms / 1000.0)
                  << "s | Độ trễ: " << seq_latency << " ms/câu | Băng thông: " << std::setprecision(1) << seq_tps << " câu/giây\n" << std::endl;

        // Bottleneck Profiling with Adaptive Performance Batching
        double p1_ms = 0.0, p2_ms = 0.0, p3_ms = 0.0;
        auto tb0 = std::chrono::high_resolution_clock::now();
        auto results = translator.translate_batch_profiled(test_sentences, 0, p1_ms, p2_ms, p3_ms);
        auto tb1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
        double b_tps = (total_runs / total_ms) * 1000.0;

        double p1_pct = (p1_ms / total_ms) * 100.0;
        double p2_pct = (p2_ms / total_ms) * 100.0;
        double p3_pct = (p3_ms / total_ms) * 100.0;

        size_t effective_batch = translator.get_adaptive_batch_size(total_runs);

        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "🔍 PHÂN TÍCH THỜI GIAN TIÊU TỐN CỦA CHẾ ĐỘ HIỆU NĂNG HIỆN TẠI (BATCH " << effective_batch << ")" << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "📊 TỔNG THỜI GIAN DỊCH (1,000 câu) : " << std::setprecision(3) << (total_ms / 1000.0) << "s | Băng thông: " << std::setprecision(1) << b_tps << " câu/s" << std::endl;
        std::cout << "   ├─ Trạm 1 (CPU Tokenizer + Trie Match) : " << std::setprecision(3) << (p1_ms / 1000.0) << "s (" << std::setprecision(1) << p1_pct << "%)" << std::endl;
        std::cout << "   ├─ Trạm 2 (GPU/NPU ONNX Tensor Run)   : " << std::setprecision(3) << (p2_ms / 1000.0) << "s (" << std::setprecision(1) << p2_pct << "%) 🔥 [TẢI PHẦN CỨNG THẬT]" << std::endl;
        std::cout << "   └─ Trạm 3 (CPU Logits & String Build) : " << std::setprecision(3) << (p3_ms / 1000.0) << "s (" << std::setprecision(1) << p3_pct << "%)\n" << std::endl;

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

        std::vector<std::string> translated_lines = translator.translate_batch(lines, 0);

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
